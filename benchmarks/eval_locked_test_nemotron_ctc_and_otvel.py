import io, os, time, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sitecustomize

import sys
sys.path.insert(0, '/dev/shm/wsp_audio/open_asr_leaderboard_repo')
from normalizer.normalizer import EnglishTextNormalizer
from harness.firewall import assert_practice_split, compute_calibration_audit_certificate
from harness.audiowsp import AcousticNuisanceProfiler
from harness.asr import ctc_greedy_decode, edit_distance_tokens
from exp2_asr_generalization.w2v2_ctc_models import load_pretrained_w2v2_ctc
from exp2_asr_generalization.nemotron_ctc_model import Nemotron35TranscribeCTC
from exp2_asr_generalization.test_ofp_scaled_dev import load_dev_utterances
from exp2_asr_generalization.eval_locked_test_nemotron_ofp import load_locked_test_utterances, official_wer
from exp2_asr_generalization.eval_locked_test_w2v2_gevd_and_compression import get_post_pos_states, pack_chunks

torch.set_grad_enabled(False)
dev = torch.device('cuda:0')
en_norm = EnglishTextNormalizer()
profiler = AcousticNuisanceProfiler(sample_rate=16000, seed=42)

assert_practice_split('dev-clean', context='eval_locked_test_nemotron_ctc_and_otvel')
assert_practice_split('dev-other', context='eval_locked_test_nemotron_ctc_and_otvel')

dc_dev = load_dev_utterances('dev_clean', 'text', max_utts=160)
do_dev = load_dev_utterances('dev_other', 'text', max_utts=160)
calib_wavs = [w for w, _ in dc_dev[:100]] + [w for w, _ in do_dev[:80]]

test_sets = {
    'test_clean':   load_locked_test_utterances('test_clean',   'text',       max_utts=200),
    'test_other':   load_locked_test_utterances('test_other',   'text',       max_utts=200),
    'tedlium_test': load_locked_test_utterances('tedlium_test', 'text',       max_utts=200),
    'mls_test':     load_locked_test_utterances('mls_test',     'transcript', max_utts=200),
}

results = {'nemotron_681m_bpe_ctc': {}, 'ot_vel_compression': {}}

# ==============================================================================
# 1. Nemotron-3.5-Transcribe 681M BPE-CTC on Locked Test Sets (OFP vs Ridge vs AdamW vs Baseline)
# ==============================================================================
print('\n==================== 1. Nemotron-3.5-Transcribe 681M BPE-CTC on Locked Test Sets ====================', flush=True)
m_nem = Nemotron35TranscribeCTC().to(dev).eval()
D = 1280; V = 23863
W0 = m_nem.ctc_head.weight.data.double().clone()
b0 = m_nem.ctc_head.bias.data.double().clone()

Hcp_list, Hp_list = [], []
for idx_u, wav in enumerate(calib_wavs):
    if idx_u % 2 == 0:
        wt = torch.from_numpy(wav).unsqueeze(0).to(dev)
        _, dec_out, _, _, _ = m_nem(wt, return_features=True)
        hc = dec_out[0].float().cpu()
        probes = profiler.generate_paired_probes(wav, split='dev-clean')
        for p_name in ('snr_noise', 'room_reverb', 'telephony_8khz', 'codec_mulaw', 'gain_shift'):
            _, pdo, _, _, _ = m_nem(torch.from_numpy(probes[p_name]).unsqueeze(0).to(dev), return_features=True)
            if pdo.size(1) == len(hc):
                Hcp_list.append(hc); Hp_list.append(pdo[0].float().cpu())

H_cpair = torch.cat(Hcp_list, dim=0).to(dev).double()
H_pert  = torch.cat(Hp_list, dim=0).to(dev).double()

Z_cpair = H_cpair @ W0.T + b0
top2 = torch.topk(Z_cpair, k=2, dim=-1).values
margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
nonblank = (Z_cpair.argmax(dim=-1) != m_nem.blank_id).double()
pi = 1.0 + 1.5 * torch.exp(-0.25 * margin) + 2.0 * nonblank
sq = torch.sqrt(pi / pi.mean()).unsqueeze(1)

mu_c = (H_cpair * sq * sq).mean(0, keepdim=True)
Hc = (H_cpair - mu_c) * sq; Hp = (H_pert - mu_c) * sq; dH = (H_pert - H_cpair) * sq
Cov_c = (Hc.T @ Hc) / len(Hc); Cov_p = (Hp.T @ Hp) / len(Hp); Cov_dH = (dH.T @ dH) / len(dH); Cross_pc = (Hp.T @ Hc) / len(Hp)
avg_eig = float(torch.trace(Cov_c)) / D
I_d = torch.eye(D, device=dev, dtype=torch.float64)
ev_c, evec_c = torch.linalg.eigh(Cov_c + 0.15 * avg_eig * I_d)
C_ih = evec_c @ torch.diag(ev_c.rsqrt()) @ evec_c.T
_, evecs_g = torch.linalg.eigh(C_ih @ Cov_dH @ C_ih)

# Cache locked test features for Nemotron 681M BPE-CTC
nem_test_cache = {}
for d_name, u_list in test_sets.items():
    c_items, n_items = [], []
    for wav, ref in u_list:
        wt = torch.from_numpy(wav).unsqueeze(0).to(dev)
        _, dec_out, _, _, _ = m_nem(wt, return_features=True)
        c_items.append((dec_out[0].double(), ref))
    for wav, ref in u_list[:50]:
        probes = profiler.generate_paired_probes(wav, split='dev-other')
        for p_type in ('snr_noise', 'room_reverb', 'telephony_8khz'):
            _, pdo, _, _, _ = m_nem(torch.from_numpy(probes[p_type]).unsqueeze(0).to(dev), return_features=True)
            n_items.append((pdo[0].double(), ref))
    nem_test_cache[d_name] = (c_items, n_items)

def eval_nem_head(W, b, label):
    wc, wn = {}, {}
    for d_name, (c_items, n_items) in nem_test_cache.items():
        wc[d_name] = round(official_wer([r for _, r in c_items], [m_nem.decode_ctc((h @ W.T + b).float()) for h, _ in c_items]) * 100.0, 2)
        wn[d_name] = round(official_wer([r for _, r in n_items], [m_nem.decode_ctc((h @ W.T + b).float()) for h, _ in n_items]) * 100.0, 2)
    mc = float(np.mean(list(wc.values())))
    mn = float(np.mean(list(wn.values())))
    all8 = 0.5 * (mc + mn)
    print(f'  {label:42s} | Clean Macro-4: {mc:5.2f}% {wc} | Nuis Macro-4: {mn:5.2f}% {wn} | Overall: {all8:5.2f}%', flush=True)
    return {'clean_macro': round(mc, 3), 'nuis_macro': round(mn, 3), 'overall_macro': round(all8, 3), 'clean': wc, 'nuis': wn}

results['nemotron_681m_bpe_ctc']['Baseline (Nemotron-3.5 681M BPE-CTC)'] = eval_nem_head(W0, b0, 'Baseline (Nemotron-3.5 681M BPE-CTC)')

# Standard Ridge on Nemotron 681M
M_ridge = torch.linalg.solve(Cov_c + 0.30 * Cov_p + 0.20 * avg_eig * I_d, Cov_c + 0.30 * Cross_pc + 0.20 * avg_eig * I_d).T
W_ridge = W0 @ M_ridge; b_ridge = b0 + (mu_c @ W0.T - mu_c @ W_ridge.T).squeeze(0)
results['nemotron_681m_bpe_ctc']['Standard Ridge (1F+0B Baseline)'] = eval_nem_head(W_ridge, b_ridge, 'Standard Ridge (1F+0B Baseline)')

# Ours: OFP-GEVD (k=128, gp=0.25, gd=0.35)
lam = 0.20 * avg_eig
M_full = torch.linalg.solve(Cov_c + 0.25 * Cov_p + 0.35 * Cov_dH + lam * I_d, Cov_c + 0.25 * Cross_pc + lam * I_d).T
U_g = C_ih @ evecs_g[:, -128:].flip(1)
Q_n, _ = torch.linalg.qr(U_g, mode='reduced')
M_op = I_d + (M_full - I_d) @ (Q_n @ Q_n.T)
W_ofp = W0 @ M_op; b_ofp = b0 + (mu_c @ W0.T - mu_c @ W_ofp.T).squeeze(0)
results['nemotron_681m_bpe_ctc']['AudioWSP + OFP-GEVD (1F+0B, Ours)'] = eval_nem_head(W_ofp, b_ofp, 'AudioWSP + OFP-GEVD (1F+0B, Ours)')

# ==============================================================================
# 2. OT-Vel + OFP vs Wanda & Magnitude at Identical Budget on Locked Test Sets
# ==============================================================================
print('\n==================== 2. OT-Vel + OFP vs Wanda & Magnitude on Locked Test Sets ====================', flush=True)
def ot_vel_prune(Z_3d, W_orig, b_orig, density, beta_d1=1.0, use_ot_gain=False):
    B, T, D_in = Z_3d.shape
    D_out = W_orig.size(0)
    Z = Z_3d.reshape(-1, D_in).float()
    W = W_orig.float(); b = b_orig.float()
    if beta_d1 > 0.0 and T > 1:
        d1_Z = (Z_3d[:, 1:] - Z_3d[:, :-1]).reshape(-1, D_in).float()
        col_sc = torch.sqrt((Z**2).mean(0) + beta_d1 * (d1_Z**2).mean(0) + 1e-8)
    else:
        col_sc = torch.sqrt((Z**2).mean(0) + 1e-8)
    k_keep = max(1, int(round(D_in * float(density))))
    idx = torch.topk(W.abs() * col_sc.unsqueeze(0), k=k_keep, dim=1).indices
    W_hat = W * torch.zeros_like(W).scatter_(1, idx, 1.0)
    if use_ot_gain:
        Zc = Z - Z.mean(0, keepdim=True)
        Yo = Zc @ W.T; Yh = Zc @ W_hat.T
        gain = (Yo.var(0, unbiased=False) / (Yh.var(0, unbiased=False) + 1e-6)).sqrt().clamp(1.0, 1.12)
        W_hat = W_hat * gain.unsqueeze(1)
    return W_hat.clone(), b.clone()

for m_key, n_keep, prot_set, density in [
    ('large_lv60_self', 22, {6,7,8,9,10,11,12,13}, 0.48),
    ('base_960h',       12, {1,2,3,4,5,6},         0.55),
]:
    teacher = load_pretrained_w2v2_ctc(m_key, dev)
    D = teacher.embed_dim
    Zt_list = [teacher(torch.from_numpy(w).unsqueeze(0).to(dev))[0][0].double().cpu() for w in calib_wavs[:100]]
    Zt_cal = torch.cat(Zt_list, dim=0).to(dev)
    top2 = torch.topk(Zt_cal, k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
    nonblank = (Zt_cal.argmax(dim=-1) != 0).double()
    pi = 1.0 + 2.0 * torch.exp(-0.25 * margin) + 2.5 * nonblank
    sq = torch.sqrt(pi / pi.mean()).unsqueeze(1)
    mu_z = (Zt_cal * sq * sq).mean(0, keepdim=True)
    Ztc = (Zt_cal - mu_z) * sq
    std_t = Zt_cal.std(0, keepdim=True).clamp(min=1e-4)

    res_m = {}
    for method in ['magnitude', 'wanda', 'ot_vel_ofp']:
        m_s = copy.deepcopy(teacher)
        m_s.compress_pos_conv_tucker_(rank_t=28)
        if n_keep < teacher.num_layers:
            m_s.wav2vec2.encoder.layers = nn.ModuleList([m_s.wav2vec2.encoder.layers[i] for i in range(n_keep)])
        s_states = get_post_pos_states(m_s, calib_wavs[:50])

        for i in range(n_keep):
            s_lyr = m_s.wav2vec2.encoder.layers[i]
            Za_s = pack_chunks([s_lyr.layer_norm(x) for x in s_states]) if s_lyr.do_stable_layer_norm else pack_chunks(s_states)
            for pname in ('q_proj', 'k_proj', 'v_proj'):
                lin = getattr(s_lyr.attention, pname)
                W_p, b_p = lin.weight.data.float(), lin.bias.data.float()
                if method == 'magnitude':
                    k_keep = max(1, int(round(W_p.size(1) * density)))
                    lin.weight.data.copy_(W_p * torch.zeros_like(W_p).scatter_(1, torch.topk(W_p.abs(), k=k_keep, dim=1).indices, 1.0))
                elif method == 'wanda':
                    Wh, _ = ot_vel_prune(Za_s, W_p, b_p, density, beta_d1=0.0, use_ot_gain=False)
                    lin.weight.data.copy_(Wh)
                else:
                    Wh, _ = ot_vel_prune(Za_s, W_p, b_p, density, beta_d1=1.0, use_ot_gain=False)
                    lin.weight.data.copy_(Wh)

            B, T, C = Za_s.shape
            att = s_lyr.attention
            H_h = att.num_heads; Dh = att.head_dim
            ctx_s = F.scaled_dot_product_attention(
                att.q_proj(Za_s).view(B,T,H_h,Dh).transpose(1,2),
                att.k_proj(Za_s).view(B,T,H_h,Dh).transpose(1,2),
                att.v_proj(Za_s).view(B,T,H_h,Dh).transpose(1,2)
            ).transpose(1,2).contiguous().view(B,T,C)
            lin_o = s_lyr.attention.out_proj
            W_op, b_op = lin_o.weight.data.float(), lin_o.bias.data.float()
            if method == 'magnitude':
                k_keep = max(1, int(round(W_op.size(1) * density)))
                lin_o.weight.data.copy_(W_op * torch.zeros_like(W_op).scatter_(1, torch.topk(W_op.abs(), k=k_keep, dim=1).indices, 1.0))
            elif method == 'wanda':
                Wh, _ = ot_vel_prune(ctx_s, W_op, b_op, density, beta_d1=0.0, use_ot_gain=False)
                lin_o.weight.data.copy_(Wh)
            else:
                Wh, _ = ot_vel_prune(ctx_s, W_op, b_op, density, beta_d1=1.0, use_ot_gain=True)
                lin_o.weight.data.copy_(Wh)

            s_mid = [x + s_lyr.attention(s_lyr.layer_norm(x)) for x in s_states] if s_lyr.do_stable_layer_norm else [s_lyr.layer_norm(x + s_lyr.attention(x)) for x in s_states]
            if i not in prot_set:
                Zf_s = pack_chunks([s_lyr.final_layer_norm(x) for x in s_mid]) if s_lyr.do_stable_layer_norm else pack_chunks(s_mid)
                lin_1 = s_lyr.feed_forward.intermediate_dense
                W1, b1 = lin_1.weight.data.float(), lin_1.bias.data.float()
                lin_2 = s_lyr.feed_forward.output_dense
                W2, b2 = lin_2.weight.data.float(), lin_2.bias.data.float()
                if method == 'magnitude':
                    k1 = max(1, int(round(W1.size(1) * density)))
                    lin_1.weight.data.copy_(W1 * torch.zeros_like(W1).scatter_(1, torch.topk(W1.abs(), k=k1, dim=1).indices, 1.0))
                    k2 = max(1, int(round(W2.size(1) * density)))
                    lin_2.weight.data.copy_(W2 * torch.zeros_like(W2).scatter_(1, torch.topk(W2.abs(), k=k2, dim=1).indices, 1.0))
                elif method == 'wanda':
                    Wh1, _ = ot_vel_prune(Zf_s, W1, b1, density, beta_d1=0.0, use_ot_gain=False)
                    lin_1.weight.data.copy_(Wh1)
                    act_s = F.gelu(lin_1(Zf_s))
                    Wh2, _ = ot_vel_prune(act_s, W2, b2, density, beta_d1=0.0, use_ot_gain=False)
                    lin_2.weight.data.copy_(Wh2)
                else:
                    Wh1, _ = ot_vel_prune(Zf_s, W1, b1, density, beta_d1=1.0, use_ot_gain=False)
                    lin_1.weight.data.copy_(Wh1)
                    act_s = F.gelu(lin_1(Zf_s))
                    Wh2, _ = ot_vel_prune(act_s, W2, b2, density, beta_d1=1.0, use_ot_gain=True)
                    lin_2.weight.data.copy_(Wh2)

            s_states = [x + s_lyr.feed_forward(s_lyr.final_layer_norm(x)) for x in s_mid] if s_lyr.do_stable_layer_norm else [s_lyr.final_layer_norm(xm + s_lyr.feed_forward(xm)) for xm in s_mid]

        n_p = sum(int((p != 0).sum().item()) for p in m_s.parameters()) + sum(b.numel() for b in m_s.buffers())
        if method == 'ot_vel_ofp':
            def get_stu_phi(wav_np):
                wt = torch.from_numpy(wav_np).unsqueeze(0).to(dev)
                _, st = m_s(wt, return_hidden_states=True)
                h0 = F.layer_norm(st[-1][0].float(), (D,)) if m_s.wav2vec2.encoder.do_stable_layer_norm else st[-1][0].float()
                h1 = F.layer_norm(st[-2][0].float(), (D,)) if m_s.wav2vec2.encoder.do_stable_layer_norm else st[-2][0].float()
                h2 = F.layer_norm(st[-3][0].float(), (D,)) if m_s.wav2vec2.encoder.do_stable_layer_norm else st[-3][0].float()
                d1 = torch.zeros_like(h0)
                if len(h0) > 2: d1[1:-1] = 0.5 * (h0[2:] - h0[:-2])
                return torch.cat([h0, h1, h2, d1], dim=-1)

            H_cal = torch.cat([get_stu_phi(w).double() for w in calib_wavs[:100]], dim=0)
            F4 = H_cal.size(1)
            mu_s = (H_cal * sq * sq).mean(0, keepdim=True)
            Hsc = (H_cal - mu_s) * sq
            Cov_s = (Hsc.T @ Hsc) / len(H_cal)
            Cross_s = (Hsc.T @ Ztc) / len(H_cal)
            lam_d = 0.03 * float(torch.trace(Cov_s)) / F4
            W_ofp = torch.linalg.solve(Cov_s + lam_d * torch.eye(F4, device=dev, dtype=torch.float64), Cross_s).T
            std_s = ((H_cal - mu_s) @ W_ofp.T).std(0, keepdim=True).clamp(min=1e-4)
            gain = (std_t / std_s).clamp(0.85, 1.35).squeeze(0)
            W_ofp = (W_ofp * gain.unsqueeze(1)).float()
            b_ofp = (mu_z.squeeze(0) - (mu_s @ W_ofp.double().T).squeeze(0)).float()
            n_p += W_ofp.numel() - m_s.lm_head.weight.numel()

            wers = {}
            for d_name, u_list in test_sets.items():
                refs = [r for _, r in u_list]
                hyps = [ctc_greedy_decode((get_stu_phi(w) @ W_ofp.T + b_ofp).cpu().numpy()) for w, _ in u_list]
                wers[d_name] = round(official_wer(refs, hyps) * 100.0, 2)
        else:
            wers = {}
            for d_name, u_list in test_sets.items():
                refs = [r for _, r in u_list]
                hyps = [ctc_greedy_decode(m_s(torch.from_numpy(w).unsqueeze(0).to(dev))[0][0].cpu().numpy()) for w, _ in u_list]
                wers[d_name] = round(official_wer(refs, hyps) * 100.0, 2)

        mc = float(np.mean(list(wers.values())))
        res_m[method] = {'params_m': round(n_p / 1e6, 2), 'macro_clean_wer': round(mc, 2), 'per_dataset': wers}
        print(f'  [{m_key:15s}] {method:16s} | {n_p/1e6:6.2f}M | Locked Test Macro-4 WER: {mc:6.2f}% | {wers}', flush=True)
    results['ot_vel_compression'][m_key] = res_m

with open('/dev/shm/wsp_audio/asr_data/nemotron_ctc_and_otvel_locked_test.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved /dev/shm/wsp_audio/asr_data/nemotron_ctc_and_otvel_locked_test.json!', flush=True)
