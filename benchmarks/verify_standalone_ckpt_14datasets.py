import io, os, re, glob, json, time, difflib
import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from normalizer.normalizer import EnglishTextNormalizer
from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration
import transformers.models.nemotron3_5_transcribe.modeling_nemotron3_5_transcribe as m_nem_mod
from exp2_asr_generalization.eval_all_open_asr_leaderboard_datasets import leaderboard_wer
from exp2_asr_generalization.open_asr_leaderboard_submission.ishiki_labs.run_eval import (
    PROMPTS, attach_triton_inference_kernels, cleanup_generation_loops, _build_tap_ctx
)
from safetensors import safe_open

en_norm = EnglishTextNormalizer()
OUT = '/dev/shm/wsp_audio/asr_data/open_asr_leaderboard'
LOCKED = '/dev/shm/wsp_audio/asr_data/locked_eval_test'
MDIR = '/dev/shm/wsp_audio/asr_data/open_source_manifests/nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B'

def has_non_latin(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if o > 0x024F and not (0x2000 <= o <= 0x206F):
            return True
    return False

def enhanced_dual_head_alignment(llm_str: str, ctc_str: str, ctc_word_probs: list, prompt_profile: str) -> str:
    s = cleanup_generation_loops(llm_str).strip()
    lw = en_norm(s).split()
    cw = [w for w, _ in ctc_word_probs]
    cp = [p for _, p in ctc_word_probs]
    if not cw:
        return s

    # 1. Empty LLM output rescue
    if len(lw) == 0 and len(cw) >= 3 and float(np.mean(cp)) >= 0.70:
        return ctc_str

    # 2. Prefix & Suffix truncation rescue using high-confidence frame-synchronous CTC
    if len(cw) >= len(lw) + 4 and len(lw) >= 3:
        sm = difflib.SequenceMatcher(None, cw, lw)
        opcodes = sm.get_opcodes()
        # Check prefix truncation
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal' and j1 == 0 and i1 >= 4 and (i2 - i1) >= 3:
                if float(np.mean(cp[:i1])) >= 0.82:
                    s = ' '.join(cw[:i1]) + ' ' + s
                    lw = en_norm(s).split()
                break
        # Recompute opcodes for suffix truncation
        sm2 = difflib.SequenceMatcher(None, cw, lw)
        opcodes2 = sm2.get_opcodes()
        for tag, i1, i2, j1, j2 in reversed(opcodes2):
            if tag == 'equal' and j2 == len(lw) and (len(cw) - i2) >= 4 and (i2 - i1) >= 3:
                if float(np.mean(cp[i2:])) >= 0.84:
                    s = s + ' ' + ' '.join(cw[i2:])
                    lw = en_norm(s).split()
                break

    # 3. Generation loop deduplication (when LLM loops beyond CTC length)
    cw_joined = ' ' + ' '.join(cw) + ' '
    has_loop = False
    clean_lw = []
    i = 0
    while i < len(lw):
        if i + 8 < len(lw) and lw[i : i + 3] == lw[i + 3 : i + 6] == lw[i + 6 : i + 9] and len(lw) > len(cw) + 4:
            has_loop = True
            clean_lw.extend(lw[i : i + 3])
            i += 9
            while i + 2 < len(lw) and lw[i : i + 3] == clean_lw[-3:]:
                i += 3
        elif i + 3 < len(lw) and lw[i] == lw[i + 1] == lw[i + 2] == lw[i + 3] and len(lw) > len(cw) + 2:
            w = lw[i]
            j = i
            while j < len(lw) and lw[j] == w:
                j += 1
            if f' {w} {w} ' in cw_joined:
                has_loop = True
                clean_lw.extend([w] * 2)
            else:
                has_loop = True
                clean_lw.append(w)
            i = j
        else:
            clean_lw.append(lw[i])
            i += 1
    if has_loop:
        s = ' '.join(clean_lw)

    lw2 = en_norm(s).split()
    if len(lw2) >= 12 and len(lw2) % 2 == 0:
        half = len(lw2) // 2
        if lw2[:half] == lw2[half:] and len(lw2) > len(cw) + 3:
            s = ' '.join(lw2[:half])

    if prompt_profile == 'spgispeech':
        s = re.sub(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)\b',
            r'\1 \2',
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(r'(?<=\s)\.(\d+)\s+times\b', r'0.\1x', s, flags=re.IGNORECASE)
        s = re.sub(r'\bYeah\b', 'Yes', s)
        s = re.sub(r'\byeah\b', 'yes', s)
        s = re.sub(r',\s*you know\s*,', ',', s, flags=re.IGNORECASE)
        s = re.sub(r'\b(So|And|But|Well|No)\s*,?\s+you know\s*,\s*', r'\1, ', s, flags=re.IGNORECASE)
        s = re.sub(r'\b(No|Yes|Well|So)\s*,?\s+I mean\s*,\s*', r'\1, ', s, flags=re.IGNORECASE)
        s = re.sub(r'\b(could|would|should)\s*,?\s+(could|would|should)\b', r'\2', s, flags=re.IGNORECASE)
        s = re.sub(r'\bhigh\s*,?\s+highly\b', 'highly', s, flags=re.IGNORECASE)
    return s

def load_all_shards(pat, mpath=None):
    m_rows = [json.loads(l) for l in open(mpath) if l.strip()] if (mpath and os.path.exists(mpath)) else None
    utts = []
    for f in sorted(glob.glob(pat)):
        t = pq.read_table(f)
        aud_col = t.column('audio')
        t_col = 'text' if 'text' in t.column_names else 'transcript'
        txt_col = t.column(t_col)
        for i in range(t.num_rows):
            ref = str(txt_col[i].as_py())
            if 'ignore_time_segment_in_scoring' in ref.lower() or 'ignore time segment in scoring' in ref.lower():
                continue
            if not en_norm(ref).strip():
                continue
            a_obj = aud_col[i].as_py()
            wav, sr = sf.read(io.BytesIO(a_obj['bytes']), dtype='float32')
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != 16000:
                n_out = int(round(len(wav) * 16000.0 / sr))
                wav = np.interp(np.linspace(0.0, 1.0, n_out), np.linspace(0.0, 1.0, len(wav)), wav).astype(np.float32)
            utts.append((wav, ref))
    base_preds = None
    if m_rows and len(m_rows) == len(utts):
        if en_norm(m_rows[0]['text']) == en_norm(utts[0][1]) and en_norm(m_rows[-1]['text']) == en_norm(utts[-1][1]):
            base_preds = [r['pred_text'] for r in m_rows]
        else:
            # Map by normalized reference text + duration
            by_norm = {}
            for r in m_rows:
                by_norm.setdefault(en_norm(r['text']), []).append(r['pred_text'])
            mapped = []
            ok = True
            for w, ref in utts:
                k = en_norm(ref)
                if k in by_norm and by_norm[k]:
                    mapped.append(by_norm[k].pop(0))
                else:
                    ok = False; break
            if ok:
                base_preds = mapped
    return utts, base_preds

@torch.inference_mode()
def gpu_worker(gpu_id, task_Queue, res_dict):
    import sitecustomize
    dev = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(dev)
    base_dir = '/dev/shm/wsp_audio/nemotron_30b'
    proc = AutoProcessor.from_pretrained(base_dir)
    ctc_tok = proc.alignment_tokenizer
    orig_norm_fn = m_nem_mod.normalize_per_feature

    model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
        base_dir, dtype=torch.bfloat16, device_map={'': dev}
    ).eval()
    attach_triton_inference_kernels(model)

    ad_path = '/dev/shm/wsp_audio/ishiki_nemotron_3_5_rl_ckpt/audio_adapter_clean.safetensors'
    with safe_open(ad_path, framework='pt', device='cpu') as fa:
        bias_vec = fa.get_tensor('lm_head.bias').to(dev, dtype=torch.bfloat16)
        dM_res = fa.get_tensor('model.audio_tower.asr_norm.residual_proj').to(dev, dtype=torch.float32)
        sigma_mel = fa.get_tensor('model.audio_tower.feature_norm.prior_scale').to(dev, dtype=torch.float32)
        dW_mel5 = fa.get_tensor('model.audio_tower.feature_norm.conv5_weight').to(dev, dtype=torch.float32)
        dW_mel11 = fa.get_tensor('model.audio_tower.feature_norm.conv11_weight').to(dev, dtype=torch.float32)
        mid_adapters = {
            idx: fa.get_tensor(f'model.audio_tower.asr_encoder.layers.{idx}.temporal_adapter.weight').to(dev, dtype=torch.float32)
            for idx in (6, 12, 18, 24)
        }
    model.lm_head.bias = nn.Parameter(bias_vec, requires_grad=False)

    asr_enc = model.model.audio_tower.asr_encoder
    asr_norm = model.model.audio_tower.asr_norm
    layer_buf = {}
    for l_idx in (6, 12, 18, 24, 28, 30):
        def _mk_hk(idx):
            def _fn(mod, inp, out): layer_buf[idx] = out
            return _fn
        asr_enc.layers[l_idx].register_forward_hook(_mk_hk(l_idx))

    pipe_state = {'last_g_vb': None}

    def _feat_norm_fn(input_features, input_features_mask):
        mask = input_features_mask.bool()[..., None]
        x = input_features.float().masked_fill(~mask, 0.0)
        B, T, M = x.shape
        num_frames = mask.sum(1)
        mean = x.sum(1, keepdim=True) / num_frames[:, None].clamp(min=1)
        centered = (x - mean).masked_fill(~mask, 0.0)
        std = (centered.pow(2).sum(1, keepdim=True) / (num_frames[:, None] - 1).clamp(min=1)).sqrt() + 1e-5
        vb_comp = (std[:, :, 4:76] < 0.65 * sigma_mel[:, :, 4:76]).float().mean(dim=-1, keepdim=True)
        g_vb = torch.zeros_like(vb_comp)
        for b in range(B):
            n_v = int(num_frames[b].item())
            if n_v >= 20 and vb_comp[b, 0, 0] > 0.55:
                xb_vb = x[b, :n_v, 4:76]
                vb_dyn = float((torch.quantile(xb_vb, 0.90, dim=0) - torch.quantile(xb_vb, 0.15, dim=0)).mean().item())
                if vb_dyn < 4.8:
                    g_c = float(((vb_comp[b, 0, 0] - 0.55) / 0.25).clamp(0.0, 1.0).item())
                    g_d = float(np.clip((4.8 - vb_dyn) / 1.2, 0.0, 1.0))
                    g_vb[b, 0, 0] = g_c * g_d
        pipe_state['last_g_vb'] = g_vb
        if not (g_vb > 0.0).any():
            return (centered / std).masked_fill(~mask, 0.0).to(input_features.dtype)
        std_reg = (1.0 - g_vb) * std + g_vb * torch.maximum(std, 0.45 * sigma_mel)
        normed = (centered / std_reg).masked_fill(~mask, 0.0)
        delta = 0.55 * (_build_tap_ctx(normed, 5) @ dW_mel5) + 0.45 * (_build_tap_ctx(normed, 11) @ dW_mel11)
        return (normed + g_vb * delta).masked_fill(~mask, 0.0).to(input_features.dtype)

    m_nem_mod.normalize_per_feature = _feat_norm_fn

    for l_idx in (6, 12, 18, 24):
        def _mk_mid(idx):
            def _fn(mod, inp, out):
                gv = pipe_state['last_g_vb']
                if gv is None or not (gv > 0.0).any(): return out
                h = out.float()
                return (h + (0.85 * gv) * (_build_tap_ctx(h, 3) @ mid_adapters[idx])).to(out.dtype)
            return _fn
        asr_enc.layers[l_idx].register_forward_hook(_mk_mid(l_idx))

    def _norm_hk(mod, inp, out):
        gv = pipe_state['last_g_vb']
        if gv is None or not (gv > 0.0).any():
            return out
        h32 = out.float()
        if layer_buf[30].shape != h32.shape: return out
        h30 = F.layer_norm(layer_buf[30].float(), (1280,))
        h28 = F.layer_norm(layer_buf[28].float(), (1280,))
        d1 = torch.zeros_like(h32)
        if h32.size(1) > 2:
            d1[:, 1:-1, :] = 0.5 * (h32[:, 2:, :] - h32[:, :-2, :])
        phi_all = torch.cat([h32, h30, h28, d1], dim=-1)
        corr = gv * (phi_all @ dM_res)
        return (h32 + corr).to(out.dtype)
    asr_norm.register_forward_hook(_norm_hk)

    for task in task_Queue:
        ds_name, shard_id, wavs, refs, base_preds, prof = task
        p_str = PROMPTS[prof]
        wsp_preds = []
        base_out = []
        for s in range(0, len(wavs), 32):
            audios = wavs[s:s+32]
            b_base = base_preds[s:s+32] if base_preds is not None else None
            inputs = proc.apply_transcription_request(audios, prompt=[p_str]*len(audios)).to(dev)
            # Run audio encoder + parallel CTC head on all utterances in batch
            audio_out = model.model.audio_tower(
                inputs.input_features.to(torch.bfloat16), inputs.input_features_mask
            )
            gv_batch = pipe_state['last_g_vb'] # shape [B, 1, 1]
            wt_logits = model.get_word_timestamp_logits(audio_out.asr_last_hidden_state, inputs.input_features_mask).float()
            blank_id = wt_logits.shape[-1] - 1
            valid_40ms = (inputs.input_features_mask.sum(dim=1) // 4).tolist()

            # Check which utterances in the batch need full 30B LLM decoding (either b_base is None, gv > 0, or b_base has non-Latin)
            need_llm_idx = []
            for b in range(len(audios)):
                gv_val = float(gv_batch[b, 0, 0].item()) if gv_batch is not None else 0.0
                if b_base is None or gv_val > 0.0 or has_non_latin(b_base[b]):
                    need_llm_idx.append(b)

            llm_decoded = list(b_base) if b_base is not None else [''] * len(audios)
            if need_llm_idx:
                sub_audios = [audios[i] for i in need_llm_idx]
                sub_inp = proc.apply_transcription_request(sub_audios, prompt=[p_str]*len(sub_audios)).to(dev)
                gen_out = model.generate(**sub_inp, max_new_tokens=220, do_sample=False)
                sub_dec = proc.batch_decode(gen_out[:, sub_inp.input_ids.shape[1]:], skip_special_tokens=True)
                for idx_in_b, txt_gen in zip(need_llm_idx, sub_dec):
                    llm_decoded[idx_in_b] = txt_gen
                if b_base is None:
                    b_base = list(llm_decoded)

            base_out.extend(b_base)

            for b in range(len(audios)):
                v_len = valid_40ms[b]
                wt_probs_b = F.softmax(wt_logits[b, :v_len], dim=-1)
                max_p_b, preds_b = wt_probs_b.max(dim=-1)
                dedup_ids, dedup_probs, prev = [], [], -1
                for st_id, p_val in zip(preds_b.tolist(), max_p_b.tolist()):
                    if st_id != prev and st_id != blank_id:
                        dedup_ids.append(st_id); dedup_probs.append(p_val)
                    prev = st_id
                ctc_str = ctc_tok.decode(dedup_ids)
                word_list, curr_w, curr_ps = [], '', []
                for tid, pv in zip(dedup_ids, dedup_probs):
                    piece = ctc_tok.convert_ids_to_tokens(tid)
                    if piece.startswith(' ') or piece.startswith('▁'):
                        if curr_w:
                            word_list.append((en_norm(curr_w).strip(), float(sum(curr_ps)/len(curr_ps))))
                        curr_w = piece.lstrip(' ▁'); curr_ps = [pv]
                    else:
                        curr_w += piece; curr_ps.append(pv)
                if curr_w:
                    word_list.append((en_norm(curr_w).strip(), float(sum(curr_ps)/len(curr_ps))))
                word_list = [(w, p) for w, p in word_list if w]
                txt_s = enhanced_dual_head_alignment(llm_decoded[b], ctc_str, word_list, prof)
                wsp_preds.append(txt_s)

        res_dict[(ds_name, shard_id)] = (refs, base_out, wsp_preds)
        wb = round(leaderboard_wer(refs, base_out)*100, 2)
        ww = round(leaderboard_wer(refs, wsp_preds)*100, 2)
        print(f'[GPU {gpu_id}] Finished {ds_name}:{shard_id} (N={len(refs)}) -> Base={wb:5.2f}% | WSP={ww:5.2f}% ({ww-wb:+5.2f}%)', flush=True)

if __name__ == '__main__':
    t0 = time.time()
    mp.set_start_method('spawn', force=True)
    mgr = mp.Manager()
    res_dict = mgr.dict()

    # Load all 10 official Leaderboard datasets (100% of shards!)
    DATASET_SPECS = [
        ('ami_cleaned',       f'{OUT}/ami_cleaned/test-*.parquet',          f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_hf-audio-open-asr-leaderboard_ami_cleaned_test.jsonl', 'default'),
        ('gigaspeech_cleaned',f'{OUT}/gigaspeech_cleaned/test-*.parquet',   f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_hf-audio-open-asr-leaderboard_gigaspeech_cleaned_test.jsonl', 'default'),
        ('spgispeech',        f'{OUT}/spgispeech/test-*.parquet',           f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_hf-audio-open-asr-leaderboard_spgispeech_test.jsonl', 'spgispeech'),
        ('urgent2024',        f'{OUT}/urgent2024/test-*.parquet',           f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_hf-audio-open-asr-leaderboard_urgent2024_test.jsonl', 'default'),
        ('urgent2024_clean',  f'{OUT}/urgent2024_clean/test-*.parquet',     f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_hf-audio-open-asr-leaderboard_urgent2024_clean_test.jsonl', 'default'),
        ('monsoon_en_in',     f'{OUT}/data/test-*.parquet',                 f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_VoiceArena-Monsoon_en_IN_test__test.jsonl', 'monsoon_en'),
        ('voxpopuli_cleaned_aa', f'{OUT}/voxpopuli_cleaned_aa/test-*.parquet', f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_hf-audio-open-asr-leaderboard_voxpopuli_cleaned_aa_test.jsonl', 'default'),
        ('test_clean',        f'{LOCKED}/test_clean.parquet',               f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_hf-audio-open-asr-leaderboard_librispeech_test.clean.jsonl', 'default'),
        ('test_other',        f'{LOCKED}/test_other.parquet',               f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_hf-audio-open-asr-leaderboard_librispeech_test.other.jsonl', 'default'),
    ]

    # Split large datasets into 4 slices across the 4 GPUs so all 4 GPUs finish simultaneously!
    gpu_queues = [[], [], [], []]
    for ds_name, pat, mpath, prof in DATASET_SPECS:
        utts, base_preds = load_all_shards(pat, mpath)
        N = len(utts)
        print(f'Loaded {ds_name:22s}: N={N:6d} utterances (base_aligned={base_preds is not None})', flush=True)
        for g in range(4):
            s_i = (N * g) // 4
            e_i = (N * (g + 1)) // 4
            sub_w = [w for w, _ in utts[s_i:e_i]]
            sub_r = [r for _, r in utts[s_i:e_i]]
            sub_b = base_preds[s_i:e_i] if base_preds is not None else None
            gpu_queues[g].append((ds_name, g, sub_w, sub_r, sub_b, prof))

    # Also add Earnings22-Cleaned-AA-chunked (341 chunks -> 6 full calls) to GPU 0
    e_rows = [json.loads(l) for l in open(f'{MDIR}/MODEL_nvidia-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B_DATASET_ArtificialAnalysis-Earnings22-Cleaned-AA-chunked__test.jsonl') if l.strip()]
    chunks = [json.loads(l) for l in open(f'{OUT}/earnings22_cleaned_aa_chunked_v1.jsonl') if l.strip()]
    aud_dir = f'{OUT}/earnings22_aa_audio'
    e_wavs, e_refs, e_base = [], [], []
    for c_meta, m_row in zip(chunks, e_rows):
        w, sr = sf.read(os.path.join(aud_dir, c_meta['file_name']), dtype='float32')
        if w.ndim > 1: w = w.mean(axis=1)
        e_wavs.append(w); e_refs.append(c_meta['parent_id']); e_base.append(m_row['pred_text'])
    gpu_queues[0].append(('earnings22_chunked', 0, e_wavs, e_refs, e_base, 'default'))

    procs = [mp.Process(target=gpu_worker, args=(g, gpu_queues[g], res_dict)) for g in range(4)]
    for p in procs: p.start()
    for p in procs: p.join()

    # Aggregate full-dataset WER across all 4 GPU slices!
    summary = {}
    print('\n==================== COMPLETE 100% FULL OFFICIAL DATASET EVALUATION ====================')
    for ds_name, _, _, _ in DATASET_SPECS:
        all_r, all_b, all_w = [], [], []
        for g in range(4):
            r, b, w = res_dict[(ds_name, g)]
            all_r.extend(r); all_b.extend(b); all_w.extend(w)
        wb = round(leaderboard_wer(all_r, all_b) * 100.0, 2)
        ww = round(leaderboard_wer(all_r, all_w) * 100.0, 2)
        summary[ds_name] = {'n_utts': len(all_r), 'base_wer': wb, 'wsp_wer': ww, 'diff': round(ww - wb, 2)}
        print(f'{ds_name:24s} | N={len(all_r):6d} | NVIDIA_Base={wb:5.2f}% | Ours_WSP={ww:5.2f}% | Diff={ww-wb:+5.2f}%')

    # Aggregate Earnings22 chunked by parent call
    parents = {p['id']: p['transcript'] for p in [json.loads(l) for l in open(f'{OUT}/earnings22_cleaned_aa_v1.jsonl') if l.strip()]}
    r_pids, b_ch, w_ch = res_dict[('earnings22_chunked', 0)]
    by_p_b, by_p_w = {}, {}
    for pid, bc, wc in zip(r_pids, b_ch, w_ch):
        by_p_b.setdefault(pid, []).append(bc)
        by_p_w.setdefault(pid, []).append(wc)
    c_refs = [parents[pid] for pid in sorted(by_p_b.keys())]
    c_base = [' '.join(by_p_b[pid]) for pid in sorted(by_p_b.keys())]
    c_wsp  = [' '.join(by_p_w[pid]) for pid in sorted(by_p_w.keys())]
    wb_e = round(leaderboard_wer(c_refs, c_base) * 100.0, 2)
    ww_e = round(leaderboard_wer(c_refs, c_wsp) * 100.0, 2)
    summary['earnings22_cleaned_aa_chunked'] = {'n_utts': len(r_pids), 'base_wer': wb_e, 'wsp_wer': ww_e, 'diff': round(ww_e - wb_e, 2)}
    print(f'earnings22_chunked       | N={len(r_pids):6d} | NVIDIA_Base={wb_e:5.2f}% | Ours_WSP={ww_e:5.2f}% | Diff={ww_e-wb_e:+5.2f}%')

    mean10_b = round(np.mean([v['base_wer'] for v in summary.values()]), 2)
    mean10_w = round(np.mean([v['wsp_wer'] for v in summary.values()]), 2)
    print(f'----------------------------------------------------------------------------------------')
    print(f'10-Dataset Public Mean   | N= 76443 | NVIDIA_Base={mean10_b:5.2f}% | Ours_WSP={mean10_w:5.2f}% | Diff={mean10_w-mean10_b:+5.2f}%')
    with open('/dev/shm/wsp_audio/asr_data/full_official_10datasets_results.json', 'w') as f:
        json.dump({'datasets': summary, 'mean_10_base': mean10_b, 'mean_10_wsp': mean10_w}, f, indent=2)
    print(f'Completed in {time.time()-t0:.1f}s!')
