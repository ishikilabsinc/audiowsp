import io, os, time, json
import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import sitecustomize

import sys
sys.path.insert(0, '/dev/shm/wsp_audio/open_asr_leaderboard_repo')
from normalizer.normalizer import EnglishTextNormalizer
from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration
from harness.firewall import compute_calibration_audit_certificate
from harness.audiowsp import AcousticNuisanceProfiler
from harness.asr import edit_distance_tokens, normalize_asr_text

torch.set_grad_enabled(False)
MODEL_DIR = '/dev/shm/wsp_audio/nemotron_30b'
BRIDGE_STATS_PATH = '/dev/shm/wsp_audio/asr_data/nemotron_bridge_stats.pt'

en_norm = EnglishTextNormalizer()
def official_wer(refs, hyps):
    tot_ed, tot_w = 0, 0
    for r, h in zip(refs, hyps):
        rw = en_norm(r).split()
        hw = en_norm(h).split()
        if not rw: continue
        tot_ed += edit_distance_tokens(rw, hw)
        tot_w += len(rw)
    return float(tot_ed / max(1, tot_w))

def strict_wer(refs, hyps):
    tot_ed, tot_w = 0, 0
    for r, h in zip(refs, hyps):
        rw = normalize_asr_text(r).split()
        hw = normalize_asr_text(h).split()
        if not rw: continue
        tot_ed += edit_distance_tokens(rw, hw)
        tot_w += len(rw)
    return float(tot_ed / max(1, tot_w))

def load_locked_test_utterances(name, text_col, max_utts=160):
    p = f'/dev/shm/wsp_audio/asr_data/locked_eval_test/{name}.parquet'
    t = pq.read_table(p)
    aud_col = t.column('audio')
    txt_col = t.column(text_col)
    utts = []
    for i in range(t.num_rows):
        if len(utts) >= max_utts: break
        ref = str(txt_col[i].as_py())
        if 'ignore_time_segment' in ref.lower(): continue
        norm_r = normalize_asr_text(ref)
        if not norm_r or len(norm_r) < 3: continue
        b_data = aud_col[i].as_py()['bytes']
        wav, sr = sf.read(io.BytesIO(b_data), dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if len(wav) < 4000 or len(wav) > 16000 * 22: continue
        utts.append((wav, ref))
    return utts

def solve_all_methods(dev):
    """Solve our OFP-4TapBridge + GEVD alongside Standard Ridge and RanPAC strictly from BRIDGE_STATS_PATH (dev_clean + dev_other)."""
    st = torch.load(BRIDGE_STATS_PATH, map_location=dev, weights_only=False)
    H_cpair   = st['H_cpair'].to(dev).double()    # (N_p, 1280)
    Phi_cpair = st['Phi_cpair'].to(dev).double()  # (N_p, 5120)
    Phi_pert  = st['Phi_pert'].to(dev).double()   # (N_p, 5120)

    D = 1280
    F_dim = 5120

    # 1. Ours: OFP-4TapBridge (Boundary-Velocity Precision Weighting + Paired Nuisance Orthogonalization + Clean Zero-Drift)
    d1 = Phi_cpair[:, 3840:]
    vel = d1.norm(dim=-1)
    pi = 1.0 + 1.5 * (vel / (vel.mean() + 1e-8)).clamp(max=3.0)
    sq = torch.sqrt(pi / pi.mean()).unsqueeze(1)

    Residual_pert = (H_cpair - Phi_pert[:, :D]) * sq
    Phic_w = Phi_cpair * sq
    Phip_w = Phi_pert * sq
    dPhi_w = (Phi_pert - Phi_cpair) * sq

    Cov_c = (Phic_w.T @ Phic_w) / len(Phic_w)
    Cov_p = (Phip_w.T @ Phip_w) / len(Phip_w)
    Cov_d = (dPhi_w.T @ dPhi_w) / len(dPhi_w)
    Cross = (Phip_w.T @ Residual_pert) / len(Phip_w)
    avg_eig = float(torch.trace(Cov_c[:D, :D])) / D

    reg = torch.ones(F_dim, device=dev, dtype=torch.float64) * (0.22 * avg_eig)
    reg[D:] = 0.90 * avg_eig
    A_ofp = Cov_c + 0.18 * Cov_p + 0.28 * Cov_d + torch.diag(reg)
    dM_ofp = torch.linalg.solve(A_ofp, 0.18 * Cross).float()  # (5120, 1280)

    # Also compute Stage-2 Speaker-Orthogonal GEVD (k=32, g=0.20) on W_proj
    from safetensors import safe_open
    with safe_open(MODEL_DIR + '/model.safetensors', framework='pt', device='cpu') as f:
        W_proj_0 = f.get_tensor('perception.proj.weight').float().to(dev).double()
        b_proj_0 = f.get_tensor('perception.proj.bias').float().to(dev).double()

    pos = torch.arange(8, dtype=torch.float64, device=dev)[:, None]
    inv_f = torch.exp(torch.arange(0, 1280, 2, dtype=torch.float64, device=dev) * -(np.log(10000.0) / 1280.0))
    S_spk = torch.zeros((8, 1280), dtype=torch.float64, device=dev)
    S_spk[:, 0::2] = torch.sin(pos * inv_f); S_spk[:, 1::2] = torch.cos(pos * inv_f)
    Q_spk, _ = torch.linalg.qr(S_spk.T, mode='reduced')
    P_orth_spk = torch.eye(1280, device=dev, dtype=torch.float64) - Q_spk @ Q_spk.T

    Hc_1d = Phi_cpair[:, :D]; Hp_1d = Phi_pert[:, :D]
    mu_c1 = Hc_1d.mean(0, keepdim=True)
    C_c1 = ((Hc_1d - mu_c1).T @ (Hc_1d - mu_c1)) / len(Hc_1d)
    C_p1 = ((Hp_1d - mu_c1).T @ (Hp_1d - mu_c1)) / len(Hp_1d)
    C_d1 = ((Hp_1d - Hc_1d).T @ (Hp_1d - Hc_1d)) / len(Hp_1d)
    Cr_pc1 = ((Hp_1d - mu_c1).T @ (Hc_1d - mu_c1)) / len(Hp_1d)
    ev1, evec1 = torch.linalg.eigh(C_c1 + 0.15 * avg_eig * torch.eye(D, device=dev, dtype=torch.float64))
    C_ih1 = evec1 @ torch.diag(ev1.rsqrt()) @ evec1.T
    _, evecs_g1 = torch.linalg.eigh(C_ih1 @ (P_orth_spk @ C_d1 @ P_orth_spk) @ C_ih1)
    U_g1 = P_orth_spk @ (C_ih1 @ evecs_g1[:, -32:].flip(1))
    Q_n1, _ = torch.linalg.qr(U_g1, mode='reduced')
    P_n1 = Q_n1 @ Q_n1.T
    I_1d = torch.eye(D, device=dev, dtype=torch.float64)
    M_full1 = torch.linalg.solve(C_c1 + 0.20 * C_p1 + 0.20 * C_d1 + 0.20 * avg_eig * I_1d, C_c1 + 0.20 * Cr_pc1 + 0.20 * avg_eig * I_1d).T
    M_sub1 = I_1d + (M_full1 - I_1d) @ P_n1
    W_proj_ofp = W_proj_0 @ M_sub1
    b_proj_ofp = b_proj_0 + (mu_c1 @ W_proj_0.T - mu_c1 @ W_proj_ofp.T).squeeze(0)

    # 2. Literature Baseline: Standard Ridge / ELM (unweighted, no clean zero-drift anchor, no dPhi penalty)
    Cov_ridge = (Phi_pert.T @ Phi_pert) / len(Phi_pert) + (0.05 * avg_eig) * torch.eye(F_dim, device=dev, dtype=torch.float64)
    Cross_ridge = (Phi_pert.T @ (H_cpair - Phi_pert[:, :D])) / len(Phi_pert)
    dM_ridge = (0.35 * torch.linalg.solve(Cov_ridge, Cross_ridge)).float()

    # 3. Literature Baseline: Iterative AdamW (20 epochs on practice bridge objective)
    with torch.enable_grad():
        dM_adam = nn.Parameter(torch.zeros((F_dim, D), device=dev, dtype=torch.float32))
        opt = torch.optim.AdamW([dM_adam], lr=2e-3, weight_decay=1e-2)
        Xp = Phi_pert.float(); Yp = (H_cpair - Phi_pert[:, :D]).float()
        Xc = Phi_cpair.float()
        bs = 2048
        for ep in range(15):
            perm = torch.randperm(len(Xp), device=dev)
            for s in range(0, len(Xp), bs):
                idx = perm[s:s+bs]
                loss = ((Xp[idx] @ dM_adam - Yp[idx])**2).mean() + 0.2 * ((Xc[idx] @ dM_adam)**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        dM_adam_final = (0.35 * dM_adam.detach()).clone()

    return {
        'Baseline (Nemotron-3.5-30B-A4B)': (None, W_proj_0.to(torch.bfloat16), b_proj_0.to(torch.bfloat16)),
        'Standard Ridge (1F+0B Baseline)': (dM_ridge, W_proj_0.to(torch.bfloat16), b_proj_0.to(torch.bfloat16)),
        'AdamW (15-Epoch Backprop)':       (dM_adam_final, W_proj_0.to(torch.bfloat16), b_proj_0.to(torch.bfloat16)),
        'AudioWSP + OFP-4TapBridge (Ours)': (dM_ofp, W_proj_0.to(torch.bfloat16), b_proj_0.to(torch.bfloat16)),
        'AudioWSP + OFP-Hybrid (Bridge+GEVD)': (dM_ofp * 0.85, W_proj_ofp.to(torch.bfloat16), b_proj_ofp.to(torch.bfloat16)),
    }

def gpu_worker(gpu_id, d_name, t_col, max_utts, return_dict):
    torch.set_grad_enabled(False)
    dev = torch.device(f'cuda:{gpu_id}')
    profiler = AcousticNuisanceProfiler(sample_rate=16000, seed=42)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map={'': dev}
    ).eval()

    audio_tower = model.model.audio_tower
    asr_enc = audio_tower.asr_encoder
    asr_norm = audio_tower.asr_norm
    mm_proj = model.model.multi_modal_projector

    layer_buf = {}
    for l_idx in (28, 30):
        def make_hook(idx):
            def fn(mod, inp, out):
                layer_buf[idx] = out
            return fn
        asr_enc.layers[l_idx].register_forward_hook(make_hook(l_idx))

    active_state = {'dM_T': None}
    def asr_norm_hook(mod, inp, out):
        dM_T = active_state['dM_T']
        if dM_T is None:
            return out
        h32 = out.float()
        h30 = F.layer_norm(layer_buf[30].float(), (1280,))
        h28 = F.layer_norm(layer_buf[28].float(), (1280,))
        if h30.shape != h32.shape:
            return out
        d1 = torch.zeros_like(h32)
        if h32.size(1) > 2:
            d1[:, 1:-1, :] = 0.5 * (h32[:, 2:, :] - h32[:, :-2, :])
        phi = torch.cat([h32, h30, h28, d1], dim=-1)
        return (h32 + phi @ dM_T).to(out.dtype)

    asr_norm.register_forward_hook(asr_norm_hook)

    u_list = load_locked_test_utterances(d_name, t_col, max_utts=max_utts)
    wavs_c = [w for w, _ in u_list]; refs_c = [r for _, r in u_list]
    wavs_n, refs_n = [], []
    for w, r in u_list[:50]:
        probes = profiler.generate_paired_probes(w, split='dev-other')
        for p_type in ('snr_noise', 'room_reverb', 'telephony_8khz'):
            wavs_n.append(probes[p_type]); refs_n.append(r)

    @torch.inference_mode()
    def transcribe_batch(wav_list, batch_size=12):
        indexed = sorted(enumerate(wav_list), key=lambda x: len(x[1]))
        out_map = {}
        for s in range(0, len(indexed), batch_size):
            batch = indexed[s:s+batch_size]
            b_wavs = [w for _, w in batch]
            inputs = processor.apply_transcription_request(b_wavs).to(dev)
            outputs = model.generate(**inputs, max_new_tokens=192, do_sample=False)
            decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for (orig_idx, _), txt in zip(batch, decoded):
                out_map[orig_idx] = txt
        return [out_map[i] for i in range(len(wav_list))]

    methods = solve_all_methods(dev)
    res = {}
    for m_name, (dM_T, W_p, b_p) in methods.items():
        active_state['dM_T'] = dM_T
        mm_proj.weight.data.copy_(W_p)
        mm_proj.bias.data.copy_(b_p)
        hyps_c = transcribe_batch(wavs_c, batch_size=12)
        hyps_n = transcribe_batch(wavs_n, batch_size=12)
        wer_c_off = official_wer(refs_c, hyps_c) * 100.0
        wer_n_off = official_wer(refs_n, hyps_n) * 100.0
        wer_c_str = strict_wer(refs_c, hyps_c) * 100.0
        wer_n_str = strict_wer(refs_n, hyps_n) * 100.0
        res[m_name] = {
            'clean_official_wer': wer_c_off,
            'nuis_official_wer': wer_n_off,
            'clean_strict_wer': wer_c_str,
            'nuis_strict_wer': wer_n_str,
            'n_clean_utts': len(wavs_c),
            'n_nuis_utts': len(wavs_n),
        }
        print(f'[GPU {gpu_id} | {d_name:12s}] {m_name:36s} -> Official Clean: {wer_c_off:5.2f}% | Nuis: {wer_n_off:5.2f}% (Strict Clean: {wer_c_str:5.2f}%)', flush=True)
    return_dict[d_name] = res

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    tasks = [
        (0, 'test_clean',   'text',       160),
        (1, 'test_other',   'text',       160),
        (2, 'tedlium_test', 'text',       160),
        (3, 'mls_test',     'transcript', 160),
    ]
    t0 = time.time()
    procs = [mp.Process(target=gpu_worker, args=(gid, dn, tc, mu, return_dict)) for gid, dn, tc, mu in tasks]
    for p in procs: p.start()
    for p in procs: p.join()

    d_names = ['test_clean', 'test_other', 'tedlium_test', 'mls_test']
    m_names = list(return_dict['test_clean'].keys())
    summary = {'datasets': d_names, 'methods': {}}

    print(f'\n==================== 4x H100 LOCKED TEST SET EVALUATION ({time.time()-t0:.1f}s) ====================')
    for m_name in m_names:
        c_off = {dn: round(return_dict[dn][m_name]['clean_official_wer'], 2) for dn in d_names}
        n_off = {dn: round(return_dict[dn][m_name]['nuis_official_wer'], 2) for dn in d_names}
        mc = float(np.mean([return_dict[dn][m_name]['clean_official_wer'] for dn in d_names]))
        mn = float(np.mean([return_dict[dn][m_name]['nuis_official_wer'] for dn in d_names]))
        all8 = 0.5 * (mc + mn)
        summary['methods'][m_name] = {
            'macro_clean_official_wer': round(mc, 3),
            'macro_nuis_official_wer': round(mn, 3),
            'macro_overall_official_wer': round(all8, 3),
            'per_dataset_clean': c_off,
            'per_dataset_nuis': n_off,
        }
        print(f'{m_name:36s} | Clean Macro-4: {mc:5.2f}% {c_off} | Nuis Macro-4: {mn:5.2f}% {n_off} | Overall: {all8:5.2f}%')

    with open('/dev/shm/wsp_audio/asr_data/nemotron_locked_test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
