import glob, io, os, re, json, time, torch
import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from safetensors import safe_open
from normalizer.normalizer import EnglishTextNormalizer
from transformers import AutoProcessor, AutoModelForMultimodalLM
from exp2_asr_generalization.eval_all_open_asr_leaderboard_datasets import leaderboard_wer

en_norm = EnglishTextNormalizer()
OUT = '/dev/shm/wsp_audio/asr_data/open_asr_leaderboard'
LOCKED = '/dev/shm/wsp_audio/asr_data/locked_eval_test'

m_snap = sorted(glob.glob('/home/ubuntu/.cache/huggingface/hub/models--Musci-research--Musci-ASR-qwen-1019-hf/snapshots/*'))[-1]
q_snap = sorted(glob.glob('/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B-hf/snapshots/*'))[-1]
o_snap = sorted(glob.glob('/home/ubuntu/.cache/huggingface/hub/models--oruk--orukqwen-asr-1.7b/snapshots/*'))[-1]

def get_tensors(snap_dir):
    out = {}
    for f in sorted(glob.glob(f'{snap_dir}/*.safetensors')):
        with safe_open(f, framework='pt', device='cpu') as sf_f:
            for k in sf_f.keys():
                out[k] = sf_f.get_tensor(k)
    return out

def build_wsp_qwen_state_dict(gamma_enc=0.50, gamma_lm=0.35):
    t_musci = get_tensors(m_snap)
    t_qwen  = get_tensors(q_snap)
    t_oruk  = get_tensors(o_snap)
    sd = {}
    for k in t_musci:
        wm = t_musci[k].float()
        wq = t_qwen[k].float()
        wo = t_oruk[k].float()
        dm = wm - wq
        do = wo - wq
        if do.norm() > 1e-5 and dm.norm() > 1e-5:
            dot_mo = torch.sum(do * dm)
            norm_m2 = torch.sum(dm * dm).clamp(min=1e-12)
            do_perp = do - (dot_mo / norm_m2) * dm
            g = gamma_enc if 'audio_tower' in k else gamma_lm
            w_new = wm + g * do_perp
            sd[k] = w_new.to(t_musci[k].dtype)
        else:
            sd[k] = t_musci[k]
    return sd

def load_full_dataset(pat):
    utts = []
    for f in sorted(glob.glob(pat)):
        t = pq.read_table(f)
        aud_col = t.column('audio')
        t_col = 'text' if 'text' in t.column_names else 'transcript'
        txt_col = t.column(t_col)
        for i in range(t.num_rows):
            r = str(txt_col[i].as_py())
            if 'ignore_time_segment_in_scoring' in r.lower() or 'ignore time segment in scoring' in r.lower(): continue
            if not en_norm(r).strip(): continue
            w, sr = sf.read(io.BytesIO(aud_col[i].as_py()['bytes']), dtype='float32')
            if w.ndim > 1: w = w.mean(axis=1)
            if sr != 16000:
                n_out = int(round(len(w) * 16000.0 / sr))
                w = np.interp(np.linspace(0.0, 1.0, n_out), np.linspace(0.0, 1.0, len(w)), w).astype(np.float32)
            utts.append((w, r))
    return utts

@torch.inference_mode()
def worker(gpu_id, tasks, ret_d):
    dev = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(dev)
    proc = AutoProcessor.from_pretrained(m_snap)
    mod = AutoModelForMultimodalLM.from_pretrained(m_snap, dtype=torch.bfloat16).to(dev).eval()
    sd = build_wsp_qwen_state_dict(gamma_enc=0.50, gamma_lm=0.35)
    mod.load_state_dict(sd, strict=False)
    mod.tie_weights()

    for ds_name, shard_id, wavs, refs in tasks:
        hyps = []
        for s in range(0, len(wavs), 32):
            b = wavs[s:s+32]
            req = proc.apply_transcription_request(audio=b, language='en').to(dev, dtype=torch.bfloat16)
            out = mod.generate(**req, max_new_tokens=220, do_sample=False)
            dec = proc.batch_decode(out[:, req['input_ids'].shape[1]:], skip_special_tokens=True)
            hyps.extend(dec)
        ret_d[(ds_name, shard_id)] = (refs, hyps)
        print(f'[GPU {gpu_id}] Finished {ds_name}:{shard_id} (N={len(refs)})', flush=True)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mgr = mp.Manager()
    ret_d = mgr.dict()
    queues = [[], [], [], []]
    for ds_name, pat in [
        ('ami_cleaned', f'{OUT}/ami_cleaned/test-*.parquet'),
        ('test_clean',  f'{LOCKED}/test_clean.parquet'),
        ('test_other',  f'{LOCKED}/test_other.parquet'),
    ]:
        utts = load_full_dataset(pat)
        N = len(utts)
        for g in range(4):
            s_i, e_i = (N * g) // 4, (N * (g + 1)) // 4
            queues[g].append((ds_name, g, [w for w, _ in utts[s_i:e_i]], [r for _, r in utts[s_i:e_i]]))

    # Add earnings22 chunked to GPU 0
    chunks = [json.loads(l) for l in open(f'{OUT}/earnings22_cleaned_aa_chunked_v1.jsonl') if l.strip()]
    aud_dir = f'{OUT}/earnings22_aa_audio'
    e_wavs, e_pids = [], []
    for c_meta in chunks:
        w, sr = sf.read(os.path.join(aud_dir, c_meta['file_name']), dtype='float32')
        if w.ndim > 1: w = w.mean(axis=1)
        e_wavs.append(w); e_pids.append(c_meta['parent_id'])
    queues[0].append(('earnings22', 0, e_wavs, e_pids))

    ps = [mp.Process(target=worker, args=(g, queues[g], ret_d)) for g in range(4)]
    for p in ps: p.start()
    for p in ps: p.join()

    for ds_name in ('ami_cleaned', 'test_clean', 'test_other'):
        all_r, all_h = [], []
        for g in range(4):
            r, h = ret_d[(ds_name, g)]
            all_r.extend(r); all_h.extend(h)
        print(f'=== {ds_name:15s} (N={len(all_r):5d}) -> WSP_Qwen_SingleModel WER = {leaderboard_wer(all_r, all_h)*100:.2f}% ===')

    parents = {p['id']: p['transcript'] for p in [json.loads(l) for l in open(f'{OUT}/earnings22_cleaned_aa_v1.jsonl') if l.strip()]}
    r_pids, h_ch = ret_d[('earnings22', 0)]
    by_p = {}
    for pid, hc in zip(r_pids, h_ch):
        by_p.setdefault(pid, []).append(hc)
    c_refs = [parents[pid] for pid in sorted(by_p.keys())]
    c_hyps = [' '.join(by_p[pid]) for pid in sorted(by_p.keys())]
    print(f'=== earnings22      (N={len(r_pids):5d}) -> WSP_Qwen_SingleModel WER = {leaderboard_wer(c_refs, c_hyps)*100:.2f}% ===')
