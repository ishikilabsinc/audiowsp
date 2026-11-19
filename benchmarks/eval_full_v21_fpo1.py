import argparse, io, json, os, shutil, sys, time, types
sys.path.insert(0, '/dev/shm/wsp_audio')
sys.path.insert(0, '/dev/shm/wsp_audio/site-packages')
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

import eval_full_dataset as efd
import eval_full_v10 as ev10
import eval_full_v18 as ev18
import scores as hs
import compressed_modules as cm
from probe_p22_compressed_4sh import load_compressed_w2v

torch.set_grad_enabled(False)
OUT = Path('/dev/shm/wsp_audio/wsp_v2_data')
TAP_12 = [0, 2, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24]

def worker_fn_v20(gpu_id, dataset_name, shard_indices, num_shards, return_dict):
    torch.set_grad_enabled(False)
    dev = torch.device(f'cuda:{gpu_id}')
    m, w2v, pos_mod, comp_layers, bridge_mod = load_compressed_w2v(dev)
    if gpu_id == 0:
        tot_p = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in bridge_mod.parameters())
        print(f'[GPU 0] {dataset_name}: Compressed Backbone Params = {tot_p:,d} ({tot_p/1e6:.2f}M vs 315.88M)!', flush=True)

    p22_npz = np.load(OUT / 'p23_fpo1_86m_sota_head.npz')
    W_wb_10S         = torch.from_numpy(p22_npz['W_wb_10S']).to(dev, dtype=torch.float32)
    b_wb_10S         = float(p22_npz['b_wb_10S'][0])
    Wq_wb_10S        = torch.from_numpy(p22_npz['Wq_wb_10S']).to(dev, dtype=torch.float32)
    bq_wb_10S        = float(p22_npz['bq_wb_10S'][0])

    W_wb_wild_10S    = torch.from_numpy(p22_npz['W_wb_wild_10S']).to(dev, dtype=torch.float32)
    b_wb_wild_10S    = float(p22_npz['b_wb_wild_10S'][0])
    Wq_wb_wild_10S   = torch.from_numpy(p22_npz['Wq_wb_wild_10S']).to(dev, dtype=torch.float32)
    bq_wb_wild_10S   = float(p22_npz['bq_wb_wild_10S'][0])

    W_pstn_10S       = torch.from_numpy(p22_npz['W_pstn_10S']).to(dev, dtype=torch.float32)
    b_pstn_10S       = float(p22_npz['b_pstn_10S'][0])
    Wq_pstn_10S      = torch.from_numpy(p22_npz['Wq_pstn_10S']).to(dev, dtype=torch.float32)
    bq_pstn_10S      = float(p22_npz['bq_pstn_10S'][0])

    W_asv5_10S       = torch.from_numpy(p22_npz['W_asv5_10S']).to(dev, dtype=torch.float32)
    b_asv5_10S       = float(p22_npz['b_asv5_10S'][0])
    Wq_asv5_10S      = torch.from_numpy(p22_npz['Wq_asv5_10S']).to(dev, dtype=torch.float32)
    bq_asv5_10S      = float(p22_npz['bq_asv5_10S'][0])

    W_inv_10S        = torch.from_numpy(p22_npz['W_inv_10S']).to(dev, dtype=torch.float32)
    b_inv_10S        = float(p22_npz['b_inv_10S'][0])

    mu_quad_10S      = torch.from_numpy(p22_npz['mu_quad_10S']).to(dev, dtype=torch.float32)
    oinv_quad_10S    = torch.from_numpy(p22_npz['oinv_quad_10S']).to(dev, dtype=torch.float32)

    w_spec_wb        = torch.from_numpy(p22_npz['w_spec_wb']).to(dev, dtype=torch.float32)
    b_spec_wb        = float(p22_npz['b_spec_wb'][0])
    w_spec_asv5      = torch.from_numpy(p22_npz['w_spec_asv5']).to(dev, dtype=torch.float32)
    b_spec_asv5      = float(p22_npz['b_spec_asv5'][0])

    wq_spec544       = torch.from_numpy(p22_npz['wq_spec544']).to(dev, dtype=torch.float32)
    bq_spec544       = float(p22_npz['bq_spec544'][0])
    mu_h544          = torch.from_numpy(p22_npz['mu_h544']).to(dev, dtype=torch.float32)
    oinv_h544        = torch.from_numpy(p22_npz['oinv_h544']).to(dev, dtype=torch.float32)

    W_rout_asv5_7S   = torch.from_numpy(p22_npz['W_rout_asv5_7S']).to(dev, dtype=torch.float32)
    w_rout_asv5_stsp = torch.from_numpy(p22_npz['w_rout_asv5_stsp']).to(dev, dtype=torch.float32)
    b_rout_asv5      = float(p22_npz['b_rout_asv5'][0])

    W_rout_wild_7S   = torch.from_numpy(p22_npz['W_rout_wild_7S']).to(dev, dtype=torch.float32)
    w_rout_wild_stsp = torch.from_numpy(p22_npz['w_rout_wild_stsp']).to(dev, dtype=torch.float32)
    b_rout_wild      = float(p22_npz['b_rout_wild'][0])

    win400 = torch.hann_window(400, device=dev)
    win160 = torch.hann_window(160, device=dev)

    def forward_crop_batch(wavs_np):
        N_c = len(wavs_np)
        out_wb_st  = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_wb_10S + bq_wb_10S
        out_wb_wi  = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_wb_wild_10S + bq_wb_wild_10S
        out_pstn_l = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_pstn_10S
        out_pstn_q = torch.zeros(N_c, device=dev, dtype=torch.float32) + bq_pstn_10S
        out_asv5_l = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_asv5_10S
        out_asv5_q = torch.zeros(N_c, device=dev, dtype=torch.float32) + bq_asv5_10S
        out_inv_l  = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_inv_10S
        out_rasv5  = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_rout_asv5
        out_rwild  = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_rout_wild
        out_ws1    = torch.zeros(N_c, device=dev, dtype=torch.float32)
        out_sh_wb  = torch.zeros(N_c, device=dev, dtype=torch.float32)
        out_sh_ps  = torch.zeros(N_c, device=dev, dtype=torch.float32)
        out_sh_a5  = torch.zeros(N_c, device=dev, dtype=torch.float32)
        out_diffe  = torch.zeros(N_c, device=dev, dtype=torch.float32)

        with torch.inference_mode():
            for i in range(0, N_c, 128):
                b = torch.from_numpy(wavs_np[i:i+128]).to(dev)
                B = b.size(0)
                b = b / (b.abs().amax(dim=1, keepdim=True).clamp(min=0.15))
                stft_f = efd.compute_c_stft_gpu(b, win400).float().clamp(-20.0, 20.0)
                spec_f = ev10.compute_spectra_stft_gpu(b, win400, win160).float().clamp(-20.0, 20.0)

                wav_frames = b[:, :201*320].view(B, 201, 320)
                frame_e = torch.log(wav_frames.pow(2).mean(dim=-1) + 1e-8)
                z_e = (frame_e - frame_e.mean(dim=1, keepdim=True)) / (frame_e.std(dim=1, keepdim=True) + 1e-4)
                w_voiced = F.softmax(2.0 * z_e, dim=1).unsqueeze(-1)
                w_pause  = F.softmax(-2.0 * z_e, dim=1).unsqueeze(-1)

                x = b.unsqueeze(1)
                for conv in w2v.feature_extractor.conv_layers:
                    x = conv[0](x)
                    x = F.layer_norm(x.transpose(-2, -1).float(), (512,), conv[2][1].weight.float(), conv[2][1].bias.float()).transpose(-2, -1).type_as(x)
                    x = F.gelu(x)
                x = x.transpose(1, 2)
                x = F.layer_norm(x.float(), (512,), w2v.layer_norm.weight.float(), w2v.layer_norm.bias.float())
                x = w2v.post_extract_proj(x)
                x = x + pos_mod(x.transpose(1, 2)).transpose(1, 2)

                cur = {0: F.layer_norm(x.float(), (1024,))}
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    for l_idx, layer in enumerate(comp_layers, start=1):
                        x = layer(x)
                        if l_idx in TAP_12:
                            cur[l_idx] = F.layer_norm(x.float(), (1024,))
                    x_br = bridge_mod(x)
                    cur[24] = F.layer_norm(x_br.float(), (1024,), w2v.encoder.layer_norm.weight.float(), w2v.encoder.layer_norm.bias.float())

                sl = slice(i, i+B)
                stsp = torch.cat([stft_f, spec_f], dim=1)
                out_rasv5[sl] += stsp @ w_rout_asv5_stsp
                out_rwild[sl] += stsp @ w_rout_wild_stsp
                for p_pos, l_idx in enumerate(TAP_12):
                    st = torch.nan_to_num(cur[l_idx], 0.0, 0.0, 0.0)
                    mu = st.mean(dim=1); sd_t = st.std(dim=1) + 1e-4; st_cmvn = (st - mu.unsqueeze(1)) / sd_t.unsqueeze(1)
                    d1 = st_cmvn[:, 1:, :] - st_cmvn[:, :-1, :]; d2 = d1[:, 1:, :] - d1[:, :-1, :]
                    s2 = st_cmvn[:, 2:, :] - st_cmvn[:, :-2, :]; s4 = st_cmvn[:, 4:, :] - st_cmvn[:, :-4, :]
                    d1_norm = d1 / (d1.std(dim=1, keepdim=True) + 1e-4)
                    vp_diff = (w_voiced * st_cmvn).sum(dim=1) - (w_pause * st_cmvn).sum(dim=1)
                    top_d1  = torch.log(torch.topk(d1.pow(2), k=30, dim=1)[0].mean(dim=1).sqrt() + 1e-4)
                    lag1_ch = (st_cmvn * torch.roll(st_cmvn, shifts=1, dims=-1)).mean(dim=1)
                    d1_ch   = (d1_norm * torch.roll(d1_norm, shifts=1, dims=-1)).mean(dim=1)
                    log_s4  = torch.log(s4.std(dim=1) + 1e-4)
                    f7_vec = torch.cat([mu.clamp(-20,20), torch.log(sd_t).clamp(-20,20), torch.log(d1.std(1)+1e-4).clamp(-20,20), torch.log(d2.std(1)+1e-4).clamp(-20,20), torch.log(s2.std(1)+1e-4).clamp(-20,20), vp_diff.clamp(-20,20), top_d1.clamp(-20,20)], dim=1)
                    f10_vec = torch.cat([f7_vec, lag1_ch.clamp(-20,20), d1_ch.clamp(-20,20), log_s4.clamp(-20,20)], dim=1)
                    z_pos = F.relu((f10_vec - mu_quad_10S[p_pos]) * oinv_quad_10S[p_pos])
                    zq    = torch.log1p(z_pos * z_pos)

                    out_rasv5[sl]  += f7_vec  @ W_rout_asv5_7S[p_pos]
                    out_rwild[sl]  += f7_vec  @ W_rout_wild_7S[p_pos]
                    out_wb_st[sl]  += (f10_vec @ W_wb_10S[p_pos])      + (zq @ Wq_wb_10S[p_pos])
                    out_wb_wi[sl]  += (f10_vec @ W_wb_wild_10S[p_pos]) + (zq @ Wq_wb_wild_10S[p_pos])
                    out_pstn_l[sl] += f10_vec @ W_pstn_10S[p_pos]
                    out_pstn_q[sl] += zq      @ Wq_pstn_10S[p_pos]
                    out_asv5_l[sl] += f10_vec @ W_asv5_10S[p_pos]
                    out_asv5_q[sl] += zq      @ Wq_asv5_10S[p_pos]
                    out_inv_l[sl]  += f10_vec @ W_inv_10S[p_pos]

                logits, hid = efd.run_aasist_from_ssl(m, cur[24])
                out_ws1[sl]   = -logits[:, 1].float()
                hst = torch.cat([hid.float().clamp(-20.0, 20.0), stft_f, spec_f], dim=1)
                qhst = torch.log1p(F.relu((hst - mu_h544) * oinv_h544).pow(2))
                sq_544 = (qhst @ wq_spec544) + bq_spec544
                sh_wb_lin = (hst @ w_spec_wb) + b_spec_wb
                out_sh_ps[sl] = sh_wb_lin
                out_sh_wb[sl] = sh_wb_lin + 0.25 * sq_544
                out_sh_a5[sl] = (hst @ w_spec_asv5) + b_spec_asv5 + 0.15 * sq_544
                out_diffe[sl] = stft_f[:, 33:62].mean(dim=1) - stft_f[:, 3:27].mean(dim=1)

        return torch.stack([out_wb_st, out_wb_wi, out_pstn_l, out_pstn_q, out_asv5_l, out_asv5_q, out_inv_l, out_rasv5, out_rwild, out_ws1, out_sh_wb, out_sh_ps, out_sh_a5, out_diffe], dim=1)

    tmp_dir = Path(f'/dev/shm/wsp_audio/tmp_worker_{gpu_id}')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    decode_pool = ThreadPoolExecutor(max_workers=8)
    prefetch_pool = ThreadPoolExecutor(max_workers=3)

    futures = {}
    for s_idx in shard_indices[:3]:
        futures[s_idx] = prefetch_pool.submit(ev18.load_and_decode_shard_3crop, dataset_name, s_idx, num_shards, tmp_dir, decode_pool)

    uids_all, labels_all, s_p22_all, s_p22h_all = [], [], [], []
    for pos, s_idx in enumerate(shard_indices):
        if pos + 3 < len(shard_indices):
            nxt = shard_indices[pos + 3]
            futures[nxt] = prefetch_pool.submit(ev18.load_and_decode_shard_3crop, dataset_name, nxt, num_shards, tmp_dir, decode_pool)
        uids, lbls, wavs1, wavs2, wavs3, has_c2, has_c3 = futures.pop(s_idx).result()

        with torch.inference_mode():
            comps = forward_crop_batch(wavs1)
            if has_c2.any():
                c2_only = has_c2 & (~has_c3)
                if c2_only.any():
                    c2_sub_mask = (~has_c3[has_c2])
                    c2_idx = torch.from_numpy(np.where(c2_only)[0]).to(dev)
                    comps2_all = forward_crop_batch(wavs2)
                    comps[c2_idx] = 0.5 * (comps[c2_idx] + comps2_all[torch.from_numpy(np.where(c2_sub_mask)[0]).to(dev)])
                    if has_c3.any():
                        c3_idx = torch.from_numpy(np.where(has_c3)[0]).to(dev)
                        c3_in_c2 = torch.from_numpy(np.where(has_c3[has_c2])[0]).to(dev)
                        comps3 = forward_crop_batch(wavs3)
                        comps[c3_idx] = (comps[c3_idx] + comps2_all[c3_in_c2] + comps3) / 3.0
                else:
                    c3_idx = torch.from_numpy(np.where(has_c3)[0]).to(dev)
                    comps2 = forward_crop_batch(wavs2)
                    comps3 = forward_crop_batch(wavs3)
                    comps[c3_idx] = (comps[c3_idx] + comps2 + comps3) / 3.0

            s_wb_st, s_wb_wi, s_pstn_l, s_pstn_q, s_asv5_l, s_asv5_q, s_inv_l, r_asv5, r_wild, w_s1, sh_wb, sh_ps, sh_asv5, diff_e = comps.unbind(dim=1)

            bf_a5 = torch.sigmoid(2.0 * (4.50 - s_inv_l))
            s_asv5_l_casn = 0.20 * s_asv5_l + 0.80 * s_inv_l - 0.50 * bf_a5 * F.relu(s_asv5_l - s_inv_l - 0.50)
            s_asv5_q_casn = (1.0 - 0.45 * bf_a5) * s_asv5_q

            bf_ps = torch.sigmoid(2.0 * (3.50 - s_inv_l))
            s_pstn_l_casn = 0.90 * s_pstn_l + 0.10 * s_inv_l - 0.35 * bf_ps * F.relu(s_pstn_l - s_inv_l - 0.50)

            g_wild = torch.sigmoid(2.0 * (r_wild - 2.20))
            g_asv5 = torch.sigmoid(4.0 * (r_asv5 - 0.45))
            g_pstn = (1.0 - g_asv5) * torch.sigmoid(12.0 * (-1.52 - diff_e))
            g_wb   = torch.clamp(1.0 - g_asv5 - g_pstn, min=0.0)

            sc_wb_sota   = (1.0 - g_wild) * (0.08 * w_s1 + 0.14 * sh_wb + s_wb_st) + g_wild * (0.16 * w_s1 + 0.18 * sh_wb + s_wb_wi)
            sc_pstn_sota = (0.10 * w_s1 + 0.16 * sh_ps   + s_pstn_l_casn + s_pstn_q) * 1.3467 - 1.9500
            sc_asv5_sota = (0.04 * w_s1 + 0.10 * sh_asv5 + s_asv5_l_casn + s_asv5_q_casn) * 1.3500 + 0.8000

            sc_wb_hyb    = (1.0 - g_wild) * ((0.06 * w_s1 + 0.10 * sh_wb + 0.88 * s_wb_st) * 1.1905) + g_wild * ((0.22 * w_s1 + 0.19 * sh_wb + 0.72 * s_wb_wi) * 1.3889)
            sc_pstn_hyb  = sc_pstn_sota
            sc_asv5_hyb  = sc_asv5_sota

            sc_p22  = g_wb * sc_wb_sota + g_pstn * sc_pstn_sota + g_asv5 * sc_asv5_sota
            sc_p22h = g_wb * sc_wb_hyb  + g_pstn * sc_pstn_hyb  + g_asv5 * sc_asv5_hyb

            s_p22_all.append(sc_p22.cpu().numpy().astype(np.float32))
            s_p22h_all.append(sc_p22h.cpu().numpy().astype(np.float32))

        uids_all.extend(uids)
        labels_all.extend(lbls)
        del wavs1, wavs2, wavs3
        print(f'[GPU {gpu_id}] {dataset_name}: shard {pos+1}/{len(shard_indices)} (shard {s_idx}) done', flush=True)

    prefetch_pool.shutdown()
    decode_pool.shutdown()
    return_dict[gpu_id] = (
        uids_all,
        np.array(labels_all, dtype=np.int64),
        np.concatenate(s_p22_all),
        np.concatenate(s_p22h_all),
    )

def eval_v20(dataset_name, num_shards):
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    ret = manager.dict()
    procs = []
    for g in range(4):
        sh_list = [s for s in range(num_shards) if s % 4 == g]
        p = ctx.Process(target=worker_fn_v20, args=(g, dataset_name, sh_list, num_shards, ret))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f'Worker exited with {p.exitcode}')

    uids, labels, s_p22, s_p22h = [], [], [], []
    for g in range(4):
        u, l, p22, p22h = ret[g]
        uids.extend(u)
        labels.append(l)
        s_p22.append(p22)
        s_p22h.append(p22h)
    labels = np.concatenate(labels)
    s_p22 = np.concatenate(s_p22)
    s_p22h = np.concatenate(s_p22h)

    out_dir = Path('/dev/shm/wsp_audio/full_results')
    prev = json.loads((out_dir / f'{dataset_name}.json').read_text())
    prev['p23_fpo1_86m_sota'] = {
        'eer_percent': round(hs.equal_error_rate(s_p22, labels) * 100, 4),
        'auc': round(hs.roc_auc(s_p22, labels), 6),
        'params': 86749578,
    }
    prev['p23_fpo1_86m_hybrid'] = {
        'eer_percent': round(hs.equal_error_rate(s_p22h, labels) * 100, 4),
        'auc': round(hs.roc_auc(s_p22h, labels), 6),
        'params': 86749578,
    }
    (out_dir / f'{dataset_name}.json').write_text(json.dumps(prev, indent=2) + '\n')
    print(f'=== {dataset_name} (n={len(labels)}) ===')
    print('  p23_fpo1_86m_sota  :', prev['p23_fpo1_86m_sota'])
    print('  p23_fpo1_86m_hybrid:', prev['p23_fpo1_86m_hybrid'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('num_shards', type=int)
    args = ap.parse_args()
    eval_v20(args.dataset, args.num_shards)
