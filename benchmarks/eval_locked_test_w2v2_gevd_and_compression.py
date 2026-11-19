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
from exp2_asr_generalization.w2v2_ctc_models import load_pretrained_w2v2_ctc, FactorizedLinear
from exp2_asr_generalization.test_ofp_scaled_dev import load_dev_utterances, text_to_ctc_ids, ctc_viterbi_align
from exp2_asr_generalization.eval_locked_test_nemotron_ofp import load_locked_test_utterances, official_wer

torch.set_grad_enabled(False)
dev = torch.device('cuda:0')
en_norm = EnglishTextNormalizer()
profiler = AcousticNuisanceProfiler(sample_rate=16000, seed=42)

def get_post_pos_states(model, wavs):
    outs = []
    for w in wavs:
        wt = torch.from_numpy(w).unsqueeze(0).to(dev)
        wt = (wt - wt.mean(dim=-1, keepdim=True)) / torch.sqrt(wt.var(dim=-1, unbiased=False, keepdim=True) + 1e-7)
        feats = model.wav2vec2.feature_extractor(wt.unsqueeze(1)).transpose(1, 2)
        h = model.wav2vec2.feature_projection(feats)
        pos = model.wav2vec2.encoder.pos_conv_embed(h)
        if not model.wav2vec2.encoder.do_stable_layer_norm:
            outs.append(model.wav2vec2.encoder.layer_norm(h + pos).detach())
        else:
            outs.append((h + pos).detach())
    return outs

def pack_chunks(seq_list, chunk_len=64):
    chunks = []
    for t in seq_list:
        T = t.size(1)
        for s in range(0, T - chunk_len + 1, chunk_len):
            chunks.append(t[:, s:s+chunk_len, :])
    return torch.cat(chunks, dim=0)

def input_cov_ot_svd(Z_3d, W_orig, b_orig, rank, beta_d1=1.5, use_ot_gain=False, rel_damp=0.01):
    B, T, D_in = Z_3d.shape
    Z = Z_3d.reshape(-1, D_in).float()
    W = W_orig.float()
    b = b_orig.float()
    mu_z = Z.mean(0, keepdim=True)
    Zc = Z - mu_z
    if beta_d1 > 0.0 and T > 1:
        d1_Z = (Z_3d[:, 1:] - Z_3d[:, :-1]).reshape(-1, D_in).float()
        Cov = (Zc.T @ Zc) / len(Z) + beta_d1 * (d1_Z.T @ d1_Z) / len(d1_Z)
    else:
        Cov = (Zc.T @ Zc) / len(Z)
    lam = rel_damp * float(torch.trace(Cov)) / D_in
    ev, evec = torch.linalg.eigh(Cov + lam * torch.eye(D_in, device=Z.device))
    ev = ev.clamp(min=lam * 0.5)
    C_half = evec @ torch.diag(ev.sqrt()) @ evec.T
    C_ihalf = evec @ torch.diag(ev.rsqrt()) @ evec.T
    U, S, Vh = torch.linalg.svd(W @ C_half, full_matrices=False)
    r = min(rank, S.numel())
    sq = S[:r].sqrt()
    Ur = (U[:, :r] * sq.unsqueeze(0)).contiguous()
    Vr = (sq.unsqueeze(1) * (Vh[:r, :] @ C_ihalf)).contiguous()
    W_hat = Ur @ Vr
    if use_ot_gain:
        Yo = Zc @ W.T
        Yh = (Zc @ Vr.T) @ Ur.T
        gain = (Yo.var(0, unbiased=False) / (Yh.var(0, unbiased=False) + 1e-6)).sqrt().clamp(1.0, 1.15)
        Ur = (Ur * gain.unsqueeze(1)).contiguous()
        W_hat = Ur @ Vr
    br = (b + ((W - W_hat) @ mu_z.T).squeeze(1)).contiguous()
    return Ur.clone(), Vr.clone(), br.clone()

def output_cov_ot_svd_fc2(H_ot_3d, Y_exact_3d, W2_orig, b2_orig, rank, beta_d1=1.5, use_ot_gain=True):
    B, T, D_in = H_ot_3d.shape
    D_out = W2_orig.size(0)
    H = H_ot_3d.reshape(-1, D_in).float()
    Y_ex = Y_exact_3d.reshape(-1, D_out).float()
    W2 = W2_orig.float()
    mu_h = H.mean(0, keepdim=True)
    Hc = H - mu_h
    mu_y = Y_ex.mean(0, keepdim=True)
    Yc = Y_ex - mu_y
    Y_raw_3d = (Hc @ W2.T).view(B, T, D_out)
    if beta_d1 > 0.0 and T > 1:
        d1_Yr = (Y_raw_3d[:, 1:] - Y_raw_3d[:, :-1]).reshape(-1, D_out)
        Cov_Y = (Y_raw_3d.reshape(-1, D_out).T @ Y_raw_3d.reshape(-1, D_out)) / len(H) + beta_d1 * (d1_Yr.T @ d1_Yr) / len(d1_Yr)
    else:
        Cov_Y = (Y_raw_3d.reshape(-1, D_out).T @ Y_raw_3d.reshape(-1, D_out)) / len(H)
    _, evec_Y = torch.linalg.eigh(Cov_Y)
    r = min(rank, D_out)
    U_y = evec_Y[:, -r:].flip(1).contiguous()
    V_y = (U_y.T @ W2).contiguous()
    P_3d = F.linear(H_ot_3d - mu_h, V_y)
    P_flat = P_3d.reshape(-1, r)
    if beta_d1 > 0.0 and T > 1:
        d1_P = (P_3d[:, 1:] - P_3d[:, :-1]).reshape(-1, r)
        d1_Yt = (Y_exact_3d[:, 1:] - Y_exact_3d[:, :-1]).reshape(-1, D_out).float()
        Cov_p = (P_flat.T @ P_flat) / len(P_flat) + beta_d1 * (d1_P.T @ d1_P) / len(d1_P)
        Cross_p = (P_flat.T @ Yc) / len(P_flat) + beta_d1 * (d1_P.T @ d1_Yt) / len(d1_P)
    else:
        Cov_p = (P_flat.T @ P_flat) / len(P_flat)
        Cross_p = (P_flat.T @ Yc) / len(P_flat)
    lam_p = 0.015 * float(torch.trace(Cov_p)) / r
    U_y_opt = torch.linalg.solve(Cov_p + lam_p * torch.eye(r, device=H.device), Cross_p + lam_p * U_y.T).T
    if use_ot_gain:
        Yh_c = P_flat @ U_y_opt.T
        gain = (Yc.var(0, unbiased=False) / (Yh_c.var(0, unbiased=False) + 1e-6)).sqrt().clamp(1.0, 1.15)
        U_y_opt = (U_y_opt * gain.unsqueeze(1)).contiguous()
    b2_opt = (mu_y.squeeze(0) - (mu_h @ V_y.T @ U_y_opt.T).squeeze(0)).contiguous()
    return U_y_opt.clone(), V_y.clone(), b2_opt.clone()

assert_practice_split('dev-clean', context='eval_locked_test_w2v2_gevd')
assert_practice_split('dev-other', context='eval_locked_test_w2v2_gevd')

dc_dev = load_dev_utterances('dev_clean', 'text', max_utts=160)
do_dev = load_dev_utterances('dev_other', 'text', max_utts=160)
calib_utts = [(w, r, 'dev-clean') for w, r in dc_dev[:100]] + [(w, r, 'dev-other') for w, r in do_dev[:80]]
calib_wavs = [w for w, _, _ in calib_utts]

test_sets = {
    'test_clean':   load_locked_test_utterances('test_clean',   'text',       max_utts=200),
    'test_other':   load_locked_test_utterances('test_other',   'text',       max_utts=200),
    'tedlium_test': load_locked_test_utterances('tedlium_test', 'text',       max_utts=200),
    'mls_test':     load_locked_test_utterances('mls_test',     'transcript', max_utts=200),
}

out_summary = {'full_scale_ofp': {}, 'compression_pareto': {}}

# PART A: Full-Scale Zero-Drift Residual OFP Bridge on Wav2Vec2-Large-LV60-Self & Base-960h
for m_key in ['large_lv60_self', 'base_960h']:
    print(f'\n==================== PART A: {m_key} (1.0x Full-Scale Zero-Drift OFP on Locked Test Sets) ====================', flush=True)
    m = load_pretrained_w2v2_ctc(m_key, dev)
    D = m.embed_dim; V = 32
    W0 = m.lm_head.weight.data.double().clone()
    b0 = m.lm_head.bias.data.double().clone()

    def extract_phi(states):
        h0 = states[-1][0].float()
        h1 = F.layer_norm(states[-2][0].float(), (D,)) * 0.30
        curv = torch.zeros_like(h0)
        if len(h0) > 2:
            curv[1:-1] = h0[1:-1] - 0.5 * (h0[:-2] + h0[2:])
        return torch.cat([h0, h1, curv], dim=-1)

    F_dim = 3 * D
    W_prior_3 = torch.zeros((V, F_dim), device=dev, dtype=torch.float64)
    W_prior_3[:, :D] = W0

    Phi_p_list, Phi_cp_list, Z0_cp_list, Z0_p_list = [], [], [], []
    for idx_u, (wav, ref, _) in enumerate(calib_utts):
        if idx_u % 2 == 0:
            wt = torch.from_numpy(wav).unsqueeze(0).to(dev)
            logits, states = m(wt, return_hidden_states=True)
            z0 = logits[0].double()
            phi = extract_phi(states).double()
            probes = profiler.generate_paired_probes(wav, split='dev-clean')
            for p_name in ('snr_noise', 'room_reverb', 'telephony_8khz', 'codec_mulaw'):
                p_logits, p_states = m(torch.from_numpy(probes[p_name]).unsqueeze(0).to(dev), return_hidden_states=True)
                p_phi = extract_phi(p_states).double()
                if len(p_phi) == len(phi):
                    Phi_p_list.append(p_phi); Phi_cp_list.append(phi); Z0_cp_list.append(z0); Z0_p_list.append(p_logits[0].double())

    Phi_p  = torch.cat(Phi_p_list, dim=0)
    Phi_cp = torch.cat(Phi_cp_list, dim=0)
    Z0_cp  = torch.cat(Z0_cp_list, dim=0)
    Z0_p   = torch.cat(Z0_p_list, dim=0)

    test_cache = {}
    for d_name, u_list in test_sets.items():
        c_items, n_items = [], []
        for wav, ref in u_list:
            wt = torch.from_numpy(wav).unsqueeze(0).to(dev)
            _, st = m(wt, return_hidden_states=True)
            c_items.append((extract_phi(st).double(), ref))
        for wav, ref in u_list[:50]:
            probes = profiler.generate_paired_probes(wav, split='dev-other')
            for p_type in ('snr_noise', 'room_reverb', 'telephony_8khz'):
                _, st_n = m(torch.from_numpy(probes[p_type]).unsqueeze(0).to(dev), return_hidden_states=True)
                n_items.append((extract_phi(st_n).double(), ref))
        test_cache[d_name] = (c_items, n_items)

    def eval_w2v_head(pred_fn, label):
        wc, wn = {}, {}
        for d_name, (c_items, n_items) in test_cache.items():
            wc[d_name] = round(official_wer([r for _, r in c_items], [ctc_greedy_decode(pred_fn(p).float().cpu().numpy()) for p, _ in c_items]) * 100.0, 2)
            wn[d_name] = round(official_wer([r for _, r in n_items], [ctc_greedy_decode(pred_fn(p).float().cpu().numpy()) for p, _ in n_items]) * 100.0, 2)
        mc = float(np.mean(list(wc.values())))
        mn = float(np.mean(list(wn.values())))
        all8 = 0.5 * (mc + mn)
        print(f'  {label:42s} | Clean Macro-4: {mc:5.2f}% {wc} | Nuis Macro-4: {mn:5.2f}% {wn} | Overall: {all8:5.2f}%', flush=True)
        return {'clean_macro': round(mc, 3), 'nuis_macro': round(mn, 3), 'overall_macro': round(all8, 3), 'clean': wc, 'nuis': wn}

    m_res = {}
    m_res['Pretrained Baseline (1.0x)'] = eval_w2v_head(lambda p: p @ W_prior_3.T + b0, 'Pretrained Baseline (1.0x)')

    d1_p = torch.zeros_like(Phi_cp); d1_p[1:] = Phi_cp[1:] - Phi_cp[:-1]
    vel = d1_p[:, :D].norm(dim=-1)
    pi_p = 1.0 + 1.5 * (vel / (vel.mean() + 1e-8)).clamp(max=3.0)
    sq_p = torch.sqrt(pi_p / pi_p.mean()).unsqueeze(1)

    Phicp_w = Phi_cp * sq_p; Phip_w = Phi_p * sq_p; dPhi_w = (Phi_p - Phi_cp) * sq_p; Res_z = (Z0_cp - Z0_p) * sq_p
    Cov_c3 = (Phicp_w.T @ Phicp_w) / len(Phicp_w)
    Cov_p3 = (Phip_w.T @ Phip_w) / len(Phip_w)
    Cov_d3 = (dPhi_w.T @ dPhi_w) / len(dPhi_w)
    Cross_res = (Phip_w.T @ Res_z) / len(Phip_w)
    avg_eig = float(torch.trace(Cov_c3[:D, :D])) / D

    for g_c, g_d in [(0.12, 0.25), (0.20, 0.35)]:
        reg = torch.ones(F_dim, device=dev, dtype=torch.float64) * (0.30 * avg_eig)
        reg[D:] = 0.90 * avg_eig
        dW_T = torch.linalg.solve(Cov_c3 + g_c * Cov_p3 + g_d * Cov_d3 + torch.diag(reg), g_c * Cross_res)
        W_ofp = W_prior_3 + dW_T.T
        m_res[f'AudioWSP + OFP-ZeroDrift(gc={g_c})'] = eval_w2v_head(lambda p, W=W_ofp, b=b0: p @ W.T + b, f'AudioWSP + OFP-ZeroDrift(gc={g_c})')

    out_summary['full_scale_ofp'][m_key] = m_res

# PART B: Compression Benchmark on Locked Test Sets
print('\n==================== PART B: Compression Benchmark on Locked Test Sets ====================', flush=True)
for m_key, n_keep, prot_set, r_a, r_f in [
    ('large_lv60_self', 22, {6,7,8,9,10,11,12,13}, 256, 256),
    ('base_960h',       12, {1,2,3,4,5,6},         256, 256),
]:
    teacher = load_pretrained_w2v2_ctc(m_key, dev)
    D = teacher.embed_dim
    FFN_D = 3072 if D == 768 else 4096

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

    comp_res = {}
    for method in ['standard_svd', 'magnitude', 'asvd', 'wanda', 'ot_svd_no_ofp', 'ot_svd_ofp']:
        m_s = copy.deepcopy(teacher)
        m_s.compress_pos_conv_tucker_(rank_t=28)
        if n_keep < teacher.num_layers:
            m_s.wav2vec2.encoder.layers = nn.ModuleList([m_s.wav2vec2.encoder.layers[i] for i in range(n_keep)])
        s_states = get_post_pos_states(m_s, calib_wavs[:50])

        for i in range(n_keep):
            s_lyr = m_s.wav2vec2.encoder.layers[i]
            t_lyr = teacher.wav2vec2.encoder.layers[i]
            Za_s = pack_chunks([s_lyr.layer_norm(x) for x in s_states]) if s_lyr.do_stable_layer_norm else pack_chunks(s_states)

            for pname in ('q_proj', 'k_proj', 'v_proj'):
                lin = getattr(s_lyr.attention, pname)
                W_p, b_p = lin.weight.data.float(), lin.bias.data.float()
                density = float(r_a * 2 * D) / float(W_p.numel())
                if method.startswith('ot_svd'):
                    Ur, Vr, br = input_cov_ot_svd(Za_s, W_p, b_p, rank=r_a, beta_d1=1.5, use_ot_gain=False)
                    setattr(s_lyr.attention, pname, FactorizedLinear(Ur, Vr, br).to(dev))
                elif method == 'asvd':
                    col_sc = torch.sqrt((Za_s.reshape(-1, D).float()**2).mean(0) + 1e-6)
                    U, S, Vh = torch.linalg.svd(W_p * col_sc.unsqueeze(0), full_matrices=False)
                    sq_s = S[:r_a].sqrt()
                    setattr(s_lyr.attention, pname, FactorizedLinear((U[:, :r_a]*sq_s).clone(), ((sq_s.unsqueeze(1)*Vh[:r_a, :])/col_sc.unsqueeze(0)).clone(), b_p.clone()).to(dev))
                elif method == 'standard_svd':
                    U, S, Vh = torch.linalg.svd(W_p, full_matrices=False)
                    sq_s = S[:r_a].sqrt()
                    setattr(s_lyr.attention, pname, FactorizedLinear((U[:, :r_a]*sq_s).clone(), (sq_s.unsqueeze(1)*Vh[:r_a, :]).clone(), b_p.clone()).to(dev))
                elif method == 'wanda':
                    act_n = torch.linalg.norm(Za_s.reshape(-1, D).float(), ord=2, dim=0)
                    k_keep = max(1, int(round(W_p.size(1) * density)))
                    idx = torch.topk(W_p.abs() * act_n.unsqueeze(0), k=k_keep, dim=1).indices
                    lin.weight.data.copy_(W_p * torch.zeros_like(W_p).scatter_(1, idx, 1.0))
                elif method == 'magnitude':
                    k_keep = max(1, int(round(W_p.size(1) * density)))
                    idx = torch.topk(W_p.abs(), k=k_keep, dim=1).indices
                    lin.weight.data.copy_(W_p * torch.zeros_like(W_p).scatter_(1, idx, 1.0))

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
            density_o = float(r_a * 2 * D) / float(W_op.numel())
            if method.startswith('ot_svd'):
                Ur, Vr, br = input_cov_ot_svd(ctx_s, W_op, b_op, rank=r_a, beta_d1=1.5, use_ot_gain=True)
                s_lyr.attention.out_proj = FactorizedLinear(Ur, Vr, br).to(dev)
            elif method == 'asvd':
                col_sc = torch.sqrt((ctx_s.reshape(-1, D).float()**2).mean(0) + 1e-6)
                U, S, Vh = torch.linalg.svd(W_op * col_sc.unsqueeze(0), full_matrices=False)
                sq_s = S[:r_a].sqrt()
                s_lyr.attention.out_proj = FactorizedLinear((U[:, :r_a]*sq_s).clone(), ((sq_s.unsqueeze(1)*Vh[:r_a, :])/col_sc.unsqueeze(0)).clone(), b_op.clone()).to(dev)
            elif method == 'standard_svd':
                U, S, Vh = torch.linalg.svd(W_op, full_matrices=False)
                sq_s = S[:r_a].sqrt()
                s_lyr.attention.out_proj = FactorizedLinear((U[:, :r_a]*sq_s).clone(), (sq_s.unsqueeze(1)*Vh[:r_a, :]).clone(), b_op.clone()).to(dev)
            elif method == 'wanda':
                act_n = torch.linalg.norm(ctx_s.reshape(-1, D).float(), ord=2, dim=0)
                k_keep = max(1, int(round(W_op.size(1) * density_o)))
                idx = torch.topk(W_op.abs() * act_n.unsqueeze(0), k=k_keep, dim=1).indices
                lin_o.weight.data.copy_(W_op * torch.zeros_like(W_op).scatter_(1, idx, 1.0))
            elif method == 'magnitude':
                k_keep = max(1, int(round(W_op.size(1) * density_o)))
                idx = torch.topk(W_op.abs(), k=k_keep, dim=1).indices
                lin_o.weight.data.copy_(W_op * torch.zeros_like(W_op).scatter_(1, idx, 1.0))

            s_mid = [x + s_lyr.attention(s_lyr.layer_norm(x)) for x in s_states] if s_lyr.do_stable_layer_norm else [s_lyr.layer_norm(x + s_lyr.attention(x)) for x in s_states]
            if i not in prot_set:
                Zf_s = pack_chunks([s_lyr.final_layer_norm(x) for x in s_mid]) if s_lyr.do_stable_layer_norm else pack_chunks(s_mid)
                Yf_exact = t_lyr.feed_forward(Zf_s)
                lin_1 = s_lyr.feed_forward.intermediate_dense
                W1, b1 = lin_1.weight.data.float(), lin_1.bias.data.float()
                lin_2 = s_lyr.feed_forward.output_dense
                W2, b2 = lin_2.weight.data.float(), lin_2.bias.data.float()
                density_f = float(r_f * (W1.size(0) + W1.size(1))) / float(W1.numel())

                if method.startswith('ot_svd'):
                    Ur1, Vr1, br1 = input_cov_ot_svd(Zf_s, W1, b1, rank=r_f, beta_d1=1.5, use_ot_gain=False)
                    s_lyr.feed_forward.intermediate_dense = FactorizedLinear(Ur1, Vr1, br1).to(dev)
                    act_s = F.gelu(s_lyr.feed_forward.intermediate_dense(Zf_s))
                    Ur2, Vr2, br2 = output_cov_ot_svd_fc2(act_s, Yf_exact, W2, b2, rank=r_f, beta_d1=1.5, use_ot_gain=True)
                    s_lyr.feed_forward.output_dense = FactorizedLinear(Ur2, Vr2, br2).to(dev)
                elif method == 'asvd':
                    col_sc1 = torch.sqrt((Zf_s.reshape(-1, D).float()**2).mean(0) + 1e-6)
                    U, S, Vh = torch.linalg.svd(W1 * col_sc1.unsqueeze(0), full_matrices=False)
                    sq_s = S[:r_f].sqrt()
                    s_lyr.feed_forward.intermediate_dense = FactorizedLinear((U[:, :r_f]*sq_s).clone(), ((sq_s.unsqueeze(1)*Vh[:r_f, :])/col_sc1.unsqueeze(0)).clone(), b1.clone()).to(dev)
                    act_s = F.gelu(s_lyr.feed_forward.intermediate_dense(Zf_s))
                    col_sc2 = torch.sqrt((act_s.reshape(-1, FFN_D).float()**2).mean(0) + 1e-6)
                    U2, S2, Vh2 = torch.linalg.svd(W2 * col_sc2.unsqueeze(0), full_matrices=False)
                    sq_s2 = S2[:r_f].sqrt()
                    s_lyr.feed_forward.output_dense = FactorizedLinear((U2[:, :r_f]*sq_s2).clone(), ((sq_s2.unsqueeze(1)*Vh2[:r_f, :])/col_sc2.unsqueeze(0)).clone(), b2.clone()).to(dev)
                elif method == 'standard_svd':
                    U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
                    sq_s = S[:r_f].sqrt()
                    s_lyr.feed_forward.intermediate_dense = FactorizedLinear((U[:, :r_f]*sq_s).clone(), (sq_s.unsqueeze(1)*Vh[:r_f, :]).clone(), b1.clone()).to(dev)
                    U2, S2, Vh2 = torch.linalg.svd(W2, full_matrices=False)
                    sq_s2 = S2[:r_f].sqrt()
                    s_lyr.feed_forward.output_dense = FactorizedLinear((U2[:, :r_f]*sq_s2).clone(), (sq_s2.unsqueeze(1)*Vh2[:r_f, :]).clone(), b2.clone()).to(dev)
                elif method == 'wanda':
                    act_n1 = torch.linalg.norm(Zf_s.reshape(-1, D).float(), ord=2, dim=0)
                    k1 = max(1, int(round(W1.size(1) * density_f)))
                    idx1 = torch.topk(W1.abs() * act_n1.unsqueeze(0), k=k1, dim=1).indices
                    lin_1.weight.data.copy_(W1 * torch.zeros_like(W1).scatter_(1, idx1, 1.0))
                    act_s = F.gelu(lin_1(Zf_s))
                    act_n2 = torch.linalg.norm(act_s.reshape(-1, FFN_D).float(), ord=2, dim=0)
                    k2 = max(1, int(round(W2.size(1) * density_f)))
                    idx2 = torch.topk(W2.abs() * act_n2.unsqueeze(0), k=k2, dim=1).indices
                    lin_2.weight.data.copy_(W2 * torch.zeros_like(W2).scatter_(1, idx2, 1.0))
                elif method == 'magnitude':
                    k1 = max(1, int(round(W1.size(1) * density_f)))
                    idx1 = torch.topk(W1.abs(), k=k1, dim=1).indices
                    lin_1.weight.data.copy_(W1 * torch.zeros_like(W1).scatter_(1, idx1, 1.0))
                    k2 = max(1, int(round(W2.size(1) * density_f)))
                    idx2 = torch.topk(W2.abs(), k=k2, dim=1).indices
                    lin_2.weight.data.copy_(W2 * torch.zeros_like(W2).scatter_(1, idx2, 1.0))

            s_states = [x + s_lyr.feed_forward(s_lyr.final_layer_norm(x)) for x in s_mid] if s_lyr.do_stable_layer_norm else [s_lyr.final_layer_norm(xm + s_lyr.feed_forward(xm)) for xm in s_mid]

        if method == 'ot_svd_ofp':
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
            n_p = sum(p.numel() for p in m_s.parameters()) + sum(b.numel() for b in m_s.buffers()) + W_ofp.numel() - m_s.lm_head.weight.numel()

            wers = {}
            for d_name, u_list in test_sets.items():
                refs = [r for _, r in u_list]
                hyps = [ctc_greedy_decode((get_stu_phi(w) @ W_ofp.T + b_ofp).cpu().numpy()) for w, _ in u_list]
                wers[d_name] = round(official_wer(refs, hyps) * 100.0, 2)
        else:
            if method in ('wanda', 'magnitude'):
                n_p = sum(int((p != 0).sum().item()) for p in m_s.parameters()) + sum(b.numel() for b in m_s.buffers())
            else:
                n_p = sum(p.numel() for p in m_s.parameters()) + sum(b.numel() for b in m_s.buffers())
            wers = {}
            for d_name, u_list in test_sets.items():
                refs = [r for _, r in u_list]
                hyps = [ctc_greedy_decode(m_s(torch.from_numpy(w).unsqueeze(0).to(dev))[0][0].cpu().numpy()) for w, _ in u_list]
                wers[d_name] = round(official_wer(refs, hyps) * 100.0, 2)

        mc = float(np.mean(list(wers.values())))
        comp_res[method] = {'params_m': round(n_p / 1e6, 2), 'macro_clean_wer': round(mc, 2), 'per_dataset': wers}
        print(f'  [{m_key:15s}] {method:16s} | {n_p/1e6:6.2f}M | Locked Test Macro-4 WER: {mc:6.2f}% | {wers}', flush=True)

    out_summary['compression_pareto'][m_key] = comp_res

with open('/dev/shm/wsp_audio/asr_data/w2v2_locked_test_full_and_compression.json', 'w') as f:
    json.dump(out_summary, f, indent=2)
print('Saved /dev/shm/wsp_audio/asr_data/w2v2_locked_test_full_and_compression.json!', flush=True)
