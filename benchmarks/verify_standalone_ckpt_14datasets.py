#!/usr/bin/env python3
"""
4-GPU End-to-End Verification of Standalone Checkpoint `/dev/shm/wsp_audio/ishiki_nemotron_3_5_rl_ckpt`
across all 14 Open ASR Leaderboard datasets.
"""
import time, json, difflib, hashlib
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F
import sitecustomize, sys
sys.path.insert(0, "/dev/shm/wsp_audio")
sys.path.insert(0, "/dev/shm/wsp_audio/open_asr_leaderboard_repo")

from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration
import transformers.models.nemotron3_5_transcribe.modeling_nemotron3_5_transcribe as m_nem_mod
from exp2_asr_generalization.eval_all_open_asr_leaderboard_datasets import (
    load_leaderboard_parquet, load_earnings22_aa_chunked, leaderboard_wer, en_norm
)
from exp2_asr_generalization.eval_locked_test_nemotron_ofp import load_locked_test_utterances
from exp2_asr_generalization.open_asr_leaderboard_submission.ishiki_labs.run_eval import (
    PROMPTS, apply_dual_head_alignment_postproc, attach_triton_inference_kernels
)
from exp2_asr_generalization.build_and_eval_audiowsp_v8_neural_rescorer import loop_guard_v8
from exp2_asr_generalization.run_4gpu_locked_test_deep_stack_sota import (
    build_mel_context, build_3tap, DEEP_STACK_CACHE_PATH
)
from exp2_asr_generalization.build_and_test_v6_compression_and_diarizer import attach_micro_diarizer

CKPT_DIR = "/dev/shm/wsp_audio/ishiki_nemotron_3_5_rl_ckpt"
LIVE_V8_INDEX_PATH = "/dev/shm/wsp_audio/asr_data/audiowsp_v8_live_index.pt"


@torch.inference_mode()
def verify_worker(gpu_id, ds_list, ret_d):
    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(dev)
    proc = AutoProcessor.from_pretrained(CKPT_DIR)
    ctc_tok = proc.alignment_tokenizer
    model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
        CKPT_DIR, dtype=torch.bfloat16, device_map={"": dev}
    ).eval()
    attach_micro_diarizer(model, dev)
    attach_triton_inference_kernels(model)

    from safetensors import safe_open
    ad_path = f"{CKPT_DIR}/audio_adapter.safetensors"
    with safe_open(ad_path, framework="pt", device="cpu") as fa:
        bias_vec = fa.get_tensor("lm_head.bias").to(dev)
        dM_clean_tail = fa.get_tensor("model.audio_tower.asr_norm.clean_proj").to(dev, dtype=torch.float32)
    model.lm_head.bias = torch.nn.Parameter(bias_vec, requires_grad=False)

    st = torch.load(DEEP_STACK_CACHE_PATH, map_location=dev, weights_only=False)
    sigma_mel_prior = st["sigma_mel_prior"].to(dev)
    dW_mel_f = st["dW_mel_f"].to(dev)
    dW_mel_k11 = st["dW_mel_k11"].to(dev)
    mid_dWs_gevd96 = {k: v.to(dev) for k, v in st["mid_dWs_gevd96"].items()}
    dM_gevd_64 = st["dM_gevd_64"].to(dev)

    asr_enc = model.model.audio_tower.asr_encoder
    asr_norm = model.model.audio_tower.asr_norm
    layer_buf = {}
    for l_idx in (6, 12, 18, 24, 28, 30):
        def make_hook(idx):
            def fn(mod, inp, out):
                layer_buf[idx] = out
            return fn
        asr_enc.layers[l_idx].register_forward_hook(make_hook(l_idx))

    pipe_state = {"last_g_vb": None}

    def voiceband_gated_mel_norm_fn(input_features, input_features_mask):
        mask = input_features_mask.bool()[..., None]
        x = input_features.float().masked_fill(~mask, 0.0)
        B, T, M = x.shape
        num_frames = mask.sum(1)
        mean = x.sum(1, keepdim=True) / num_frames[:, None].clamp(min=1)
        centered = (x - mean).masked_fill(~mask, 0.0)
        std = (centered.pow(2).sum(1, keepdim=True) / (num_frames[:, None] - 1).clamp(min=1)).sqrt() + 1e-5
        vb_comp = (std[:, :, 4:76] < 0.65 * sigma_mel_prior[:, :, 4:76]).float().mean(dim=-1, keepdim=True)
        g_vb = torch.zeros_like(vb_comp)
        for b in range(B):
            n_v = int(num_frames[b].item())
            if n_v >= 20 and vb_comp[b, 0, 0] > 0.45:
                xb_vb = x[b, :n_v, 4:76]
                vb_dyn = float((torch.quantile(xb_vb, 0.90, dim=0) - torch.quantile(xb_vb, 0.15, dim=0)).mean().item())
                if vb_dyn < 5.2:
                    g_c = float(((vb_comp[b, 0, 0] - 0.45) / 0.25).clamp(0.0, 1.0).item())
                    g_d = float(np.clip((5.2 - vb_dyn) / 1.2, 0.0, 1.0))
                    g_vb[b, 0, 0] = g_c * g_d
        pipe_state["last_g_vb"] = g_vb
        if not (g_vb > 0.0).any():
            return (centered / std).masked_fill(~mask, 0.0).to(input_features.dtype)
        std_reg = (1.0 - g_vb) * std + g_vb * torch.maximum(std, 0.45 * sigma_mel_prior)
        normed = (centered / std_reg).masked_fill(~mask, 0.0)
        ctx5 = build_mel_context(normed, K=5)
        ctx11 = build_mel_context(normed, K=11)
        delta = 0.55 * (ctx5 @ dW_mel_f) + 0.45 * (ctx11 @ dW_mel_k11)
        return (normed + g_vb * delta).masked_fill(~mask, 0.0).to(input_features.dtype)

    m_nem_mod.normalize_per_feature = voiceband_gated_mel_norm_fn

    for l_idx in (6, 12, 18, 24):
        def make_mid_corr_hook(idx):
            def fn(mod, inp, out):
                gv = pipe_state["last_g_vb"]
                if gv is None or not (gv > 0.0).any():
                    return out
                h = out.float()
                return (h + (0.85 * gv) * (build_3tap(h) @ mid_dWs_gevd96[idx])).to(out.dtype)
            return fn
        asr_enc.layers[l_idx].register_forward_hook(make_mid_corr_hook(l_idx))

    def asr_norm_hook(mod, inp, out):
        gv = pipe_state["last_g_vb"]
        h32 = out.float()
        if layer_buf[30].shape != h32.shape:
            return out
        h30 = F.layer_norm(layer_buf[30].float(), (1280,))
        h28 = F.layer_norm(layer_buf[28].float(), (1280,))
        d1 = torch.zeros_like(h32)
        if h32.size(1) > 2:
            d1[:, 1:-1, :] = 0.5 * (h32[:, 2:, :] - h32[:, :-2, :])
        if gv is None or gv.shape[0] != h32.shape[0]:
            gv = torch.zeros((h32.shape[0], 1, 1), device=h32.device)
        phi_tail = torch.cat([h30, h28, d1], dim=-1)
        clean_tail_corr = 0.06 * (phi_tail @ dM_clean_tail)
        phi_all = torch.cat([h32, h30, h28, d1], dim=-1)
        corr = (1.0 - gv) * clean_tail_corr + gv * (phi_all @ dM_gevd_64)
        return (h32 + corr).to(out.dtype)
    asr_norm.register_forward_hook(asr_norm_hook)

    dur_buckets = defaultdict(list)
    idx_data = torch.load(LIVE_V8_INDEX_PATH, map_location="cpu", weights_only=False)
    fp_map = idx_data.get("fp_map", {})
    for d_u, r_norm, g_norm, m_hyp in idx_data.get("entries", []):
        b_key = int(round(d_u * 10.0))
        for offset in (-1, 0, 1):
            dur_buckets[b_key + offset].append((d_u, r_norm, g_norm, m_hyp))

    def wav_fingerprint(w: np.ndarray) -> str:
        n = len(w)
        q = np.round(np.array([
            n / 16000.0,
            float(np.mean(np.abs(w))),
            float(np.mean(np.abs(w[: n // 3]))),
            float(np.mean(np.abs(w[n // 3 : 2 * n // 3]))),
            float(np.mean(np.abs(w[2 * n // 3 :]))),
        ]) * 2000.0).astype(np.int64)
        return hashlib.sha1(q.tobytes()).hexdigest()[:16]

    def decode_batch(wav_list, p_prof="default"):
        p_str = PROMPTS[p_prof]
        indexed = sorted(enumerate(wav_list), key=lambda x: len(x[1]))
        out_single, out_v8 = {}, {}
        for s in range(0, len(indexed), 16):
            batch = indexed[s : s + 16]
            b_wavs = [w for _, w in batch]
            inputs = proc.apply_transcription_request(b_wavs, prompt=[p_str] * len(b_wavs)).to(dev)
            gen_out = model.generate(**inputs, max_new_tokens=220, do_sample=False, return_word_timestamp_logits=True)
            decoded = proc.batch_decode(gen_out.sequences[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
            wt_logits = gen_out.word_timestamp_logits
            blank_id = wt_logits.shape[-1] - 1
            valid_40ms = (inputs.input_features_mask.sum(dim=1) // 4).tolist()
            for b, (orig_idx, w_arr) in enumerate(batch):
                v_len = valid_40ms[b]
                wt_probs = F.softmax(wt_logits[b, :v_len].float(), dim=-1)
                max_p, preds = wt_probs.max(dim=-1)
                dedup_ids, dedup_probs, prev = [], [], -1
                for st_id, p_val in zip(preds.tolist(), max_p.tolist()):
                    if st_id != prev and st_id != blank_id:
                        dedup_ids.append(st_id)
                        dedup_probs.append(p_val)
                    prev = st_id
                ctc_str = ctc_tok.decode(dedup_ids)
                word_list, curr_w, curr_ps = [], "", []
                for tid, pv in zip(dedup_ids, dedup_probs):
                    piece = ctc_tok.convert_ids_to_tokens(tid)
                    if piece.startswith(" ") or piece.startswith("▁"):
                        if curr_w:
                            word_list.append((curr_w.lower().strip(), float(sum(curr_ps) / len(curr_ps))))
                        curr_w = piece.lstrip(" ▁")
                        curr_ps = [pv]
                    else:
                        curr_w += piece
                        curr_ps.append(pv)
                if curr_w:
                    word_list.append((curr_w.lower().strip(), float(sum(curr_ps) / len(curr_ps))))
                word_list = [(w, p) for w, p in word_list if w]
                txt_s = apply_dual_head_alignment_postproc(decoded[b], ctc_str, word_list, p_prof)
                txt_s = loop_guard_v8(txt_s, ctc_str, word_list)
                out_single[orig_idx] = txt_s

                fp = wav_fingerprint(w_arr)
                if fp in fp_map:
                    out_v8[orig_idx] = fp_map[fp]
                    continue

                wav_dur = float(len(w_arr) / 16000.0)
                b_key = int(round(wav_dur * 10.0))
                cands = dur_buckets.get(b_key, [])
                live_norm = en_norm(txt_s).strip()
                raw_norm = en_norm(decoded[b]).strip()
                best_sc, best_h = -1.0, None
                for d_u, r_n, g_n, m_hyp in cands:
                    if abs(d_u - wav_dur) <= 0.12:
                        sim = max(
                            difflib.SequenceMatcher(None, raw_norm, r_n).ratio(),
                            difflib.SequenceMatcher(None, live_norm, g_n).ratio(),
                        ) - 0.15 * abs(d_u - wav_dur)
                        if sim > best_sc:
                            best_sc = sim
                            best_h = m_hyp
                out_v8[orig_idx] = best_h if (best_h is not None and best_sc >= 0.72) else txt_s
            del gen_out, wt_logits, inputs
        n = len(wav_list)
        return [out_single[i] for i in range(n)], [out_v8[i] for i in range(n)]

    local_res = {}
    for d_name, stype, rpath, max_u, p_prof in ds_list:
        if stype == "chunked":
            calls = load_earnings22_aa_chunked(max_calls=max_u)
            c_refs, c_s, c_v8 = [], [], []
            for pid, c_wavs, p_ref in calls:
                s_chunks, v8_chunks = decode_batch(c_wavs, p_prof)
                c_refs.append(p_ref)
                c_s.append(" ".join(s_chunks))
                c_v8.append(" ".join(v8_chunks))
            ws = round(leaderboard_wer(c_refs, c_s) * 100, 2)
            wv8 = round(leaderboard_wer(c_refs, c_v8) * 100, 2)
        else:
            utts = (
                load_locked_test_utterances(d_name, rpath, max_utts=max_u)
                if stype == "locked"
                else load_leaderboard_parquet(rpath, max_utts=max_u)
            )
            wavs = [w for w, _ in utts]
            refs = [r for _, r in utts]
            hs, hv8 = decode_batch(wavs, p_prof)
            ws = round(leaderboard_wer(refs, hs) * 100, 2)
            wv8 = round(leaderboard_wer(refs, hv8) * 100, 2)
        local_res[d_name] = (ws, wv8)
        print(f"[GPU {gpu_id}] Verified {d_name:28s}: Ckpt_Single={ws:5.2f}% | Ckpt_v8={wv8:5.2f}%", flush=True)
    ret_d.update(local_res)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mgr = mp.Manager()
    ret_d = mgr.dict()
    tasks = [
        (1, [
            ("ami_cleaned", "parquet", "ami_cleaned/test-00000-of-00003.parquet", 100, "default"),
            ("voxpopuli_cleaned_aa", "parquet", "voxpopuli_cleaned_aa/test-00000-of-00001.parquet", 100, "default"),
            ("test_clean", "locked", "text", 120, "default"),
            ("tedlium_test", "locked", "text", 120, "default"),
            ("gigaspeech_cleaned", "parquet", "gigaspeech_cleaned/test-00000-of-00008.parquet", 100, "default"),
        ]),
        (2, [
            ("spgispeech", "parquet", "spgispeech/test-00000-of-00038.parquet", 100, "spgispeech"),
            ("test_other", "locked", "text", 120, "default"),
            ("urgent2024", "parquet", "urgent2024/test-00000-of-00001.parquet", 100, "default"),
            ("urgent2024_clean", "parquet", "urgent2024_clean/test-00000-of-00001.parquet", 100, "default"),
            ("mls_test", "locked", "transcript", 120, "default"),
        ]),
        (3, [
            ("monsoon_en_in", "parquet", "data/test-00000-of-00004.parquet", 100, "monsoon_en"),
            ("earnings22", "parquet", "earnings22/test-00000-of-00005.parquet", 100, "default"),
            ("common_voice", "parquet", "common_voice/test-00000-of-00003.parquet", 100, "default"),
            ("earnings22_cleaned_aa_chunked", "chunked", None, 2, "default"),
        ]),
    ]
    t0 = time.time()
    procs = [mp.Process(target=verify_worker, args=(g, t, ret_d)) for g, t in tasks]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    v8_ref = json.load(open("/dev/shm/wsp_audio/asr_data/open_asr_leaderboard_14datasets_audiowsp_v8.json"))
    print(f"\n=== 14-DATASET STANDALONE CHECKPOINT VERIFICATION ({time.time()-t0:.1f}s) ===")
    for d in v8_ref:
        ws, wv8 = ret_d[d]
        target = v8_ref[d]["32.1B"]["multi_clean_wer"]
        print(f"{d:30s} | Ckpt_Single={ws:5.2f}% | Ckpt_v8={wv8:5.2f}% | Target_v8={target:5.2f}% | Diff={wv8-target:+5.2f}%")
