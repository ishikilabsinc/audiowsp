#!/usr/bin/env python3
"""
Standalone Inference Engine for `ishiki-labs/ishiki-deepfake-0.5`.

Supports both published checkpoints in the repository:
  - `variant="86.8M"` (default): `model.safetensors` (`86.8M` parameters, `354 MB`)
  - `variant="318M"`: `model_318m_full.safetensors` (`318M` parameters, `1.27 GB` full model)
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "runtime"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import eval_full_dataset as efd
import eval_full_v10 as ev10
import compressed_modules as cm

TAP_12 = [0, 2, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24]


def create_predictor(
    ckpt_dir_or_id: str = "ishiki-labs/ishiki-deepfake-0.5",
    device: str = "cuda:0",
    batch_size: int = 32,
    variant: str = "86.8M",
):
    dev = torch.device(device)
    is_full = str(variant).upper() in ("318M", "317.8M", "FULL")
    sf_filename = "model_318m_full.safetensors" if is_full else "model.safetensors"

    p_dir = Path(ckpt_dir_or_id)
    if p_dir.is_dir():
        cfg_path = p_dir / "config.json"
        sf_path = p_dir / sf_filename
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_id=ckpt_dir_or_id, filename="config.json")
        sf_path = hf_hub_download(repo_id=ckpt_dir_or_id, filename=sf_filename)

    cfg = json.load(open(cfg_path))
    st = load_file(str(sf_path), device="cpu")

    fairseq = types.ModuleType("fairseq")
    cu = types.ModuleType("fairseq.checkpoint_utils")
    cu.load_model_ensemble_and_task = lambda *a, **k: ([efd.FastFairseqWav2Vec2()], None, None)
    fairseq.checkpoint_utils = cu
    sys.modules["fairseq"] = fairseq
    sys.modules["fairseq.checkpoint_utils"] = cu
    import ssl_model

    m = ssl_model.Model(None, dev).to(dev).eval()
    if is_full:
        m_sd = {k[len("model.") :]: v.to(dev) for k, v in st.items() if k.startswith("model.")}
        m.load_state_dict(m_sd, strict=False)
        m_params = dict(m.named_parameters())
        for k, v in m_sd.items():
            if k in m_params:
                m_params[k].data.copy_(v)
        w2v = m.ssl_model.model
        pos_mod = None
        comp_layers = None
        bridge_mod = None
    else:
        m_sd = {k[len("aasist.") :]: v.to(dev) for k, v in st.items() if k.startswith("aasist.")}
        m.load_state_dict(m_sd, strict=False)
        w2v = m.ssl_model.model
        b_sd = {k[len("backbone.") :]: v.to(dev) for k, v in st.items() if k.startswith("backbone.")}
        w2v.load_state_dict(b_sd, strict=False)
        w2v_params = dict(w2v.named_parameters())
        for k, v in b_sd.items():
            if k in w2v_params:
                w2v_params[k].data.copy_(v)

        pos_mod = cm.TuckerPosConv1d(st["pos_conv.core"], st["pos_conv.U_time"], st["pos_conv.bias"]).to(dev).eval()
        comp_layers = nn.ModuleList()
        for l_i in range(22):
            H_keep, _, _ = cfg["attn_cfg"][l_i]
            K_f, _ = cfg["ffn_cfg"][l_i]
            p = f"layers.{l_i}."
            ln1_w, ln1_b = st[p + "ln1_w"], st[p + "ln1_b"]
            ln2_w, ln2_b = st[p + "ln2_w"], st[p + "ln2_b"]
            qk_f = (st[p + "qk_W"], st[p + "qk_b"]) if (p + "qk_W") in st else (st[p + "qk_U"], st[p + "qk_V"], st[p + "qk_b"])
            v_f = (st[p + "v_W"], st[p + "v_b"]) if (p + "v_W") in st else (st[p + "v_U"], st[p + "v_V"], st[p + "v_b"])
            o_f = (st[p + "o_W"], st[p + "o_b"]) if (p + "o_W") in st else (st[p + "o_U"], st[p + "o_V"], st[p + "o_b"])
            if K_f >= 4096:
                ffn_mode = "full"
                ffn_f = (st[p + "fc1_W"], st[p + "fc1_b"], st[p + "fc2_W"], st[p + "fc2_b"])
            elif K_f == 0:
                ffn_mode = "lin"
                ffn_f = (st[p + "lin_U"], st[p + "lin_V"], st[p + "lin_b"])
            else:
                ffn_mode = "topk_res"
                ffn_f = (st[p + "fc1_W"], st[p + "fc1_b"], st[p + "fc2_W"], st[p + "res_U"], st[p + "res_V"], st[p + "comb_b"])
            comp_layers.append(cm.CompressedTransformerLayer(ln1_w, ln1_b, H_keep, qk_f, v_f, o_f, ln2_w, ln2_b, ffn_mode, ffn_f).to(dev).eval())

        bridge_mod = cm.Rank64Bridge(st["bridge.U"], st["bridge.V"], st["bridge.b"]).to(dev).eval()
        w2v.encoder.pos_conv = pos_mod
        w2v.encoder.layers = comp_layers

    def hd(k):
        return st[f"head.{k}"].to(dev, dtype=torch.float32)

    def hdf(k):
        return float(st[f"head.{k}"].reshape(-1)[0].item())

    W_wb_10S, b_wb_10S, Wq_wb_10S, bq_wb_10S = hd("W_wb_10S"), hdf("b_wb_10S"), hd("Wq_wb_10S"), hdf("bq_wb_10S")
    W_wb_wild_10S, b_wb_wild_10S, Wq_wb_wild_10S, bq_wb_wild_10S = hd("W_wb_wild_10S"), hdf("b_wb_wild_10S"), hd("Wq_wb_wild_10S"), hdf("bq_wb_wild_10S")
    W_pstn_10S, b_pstn_10S, Wq_pstn_10S, bq_pstn_10S = hd("W_pstn_10S"), hdf("b_pstn_10S"), hd("Wq_pstn_10S"), hdf("bq_pstn_10S")
    W_asv5_10S, b_asv5_10S, Wq_asv5_10S, bq_asv5_10S = hd("W_asv5_10S"), hdf("b_asv5_10S"), hd("Wq_asv5_10S"), hdf("bq_asv5_10S")
    W_inv_10S, b_inv_10S = hd("W_inv_10S"), hdf("b_inv_10S")
    mu_quad_10S, oinv_quad_10S = hd("mu_quad_10S"), hd("oinv_quad_10S")
    w_spec_wb, b_spec_wb = hd("w_spec_wb"), hdf("b_spec_wb")
    w_spec_asv5, b_spec_asv5 = hd("w_spec_asv5"), hdf("b_spec_asv5")
    wq_spec544, bq_spec544, mu_h544, oinv_h544 = hd("wq_spec544"), hdf("bq_spec544"), hd("mu_h544"), hd("oinv_h544")
    W_rout_asv5_7S, w_rout_asv5_stsp, b_rout_asv5 = hd("W_rout_asv5_7S"), hd("w_rout_asv5_stsp"), hdf("b_rout_asv5")
    W_rout_wild_7S, w_rout_wild_stsp, b_rout_wild = hd("W_rout_wild_7S"), hd("w_rout_wild_stsp"), hdf("b_rout_wild")

    win400 = torch.hann_window(400, device=dev)
    win160 = torch.hann_window(160, device=dev)

    @torch.inference_mode()
    def forward_crop_batch(wavs_np):
        N_c = len(wavs_np)
        out_wb_st = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_wb_10S + bq_wb_10S
        out_wb_wi = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_wb_wild_10S + bq_wb_wild_10S
        out_pstn_l = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_pstn_10S
        out_pstn_q = torch.zeros(N_c, device=dev, dtype=torch.float32) + bq_pstn_10S
        out_asv5_l = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_asv5_10S
        out_asv5_q = torch.zeros(N_c, device=dev, dtype=torch.float32) + bq_asv5_10S
        out_inv_l = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_inv_10S
        out_rasv5 = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_rout_asv5
        out_rwild = torch.zeros(N_c, device=dev, dtype=torch.float32) + b_rout_wild
        out_ws1 = torch.zeros(N_c, device=dev, dtype=torch.float32)
        out_sh_wb = torch.zeros(N_c, device=dev, dtype=torch.float32)
        out_sh_ps = torch.zeros(N_c, device=dev, dtype=torch.float32)
        out_sh_a5 = torch.zeros(N_c, device=dev, dtype=torch.float32)
        out_diffe = torch.zeros(N_c, device=dev, dtype=torch.float32)

        for i in range(0, N_c, batch_size):
            b = torch.from_numpy(wavs_np[i : i + batch_size]).to(dev)
            B = b.size(0)
            b = b / (b.abs().amax(dim=1, keepdim=True).clamp(min=0.15))
            stft_f = efd.compute_c_stft_gpu(b, win400).float().clamp(-20.0, 20.0)
            spec_f = ev10.compute_spectra_stft_gpu(b, win400, win160).float().clamp(-20.0, 20.0)

            wav_frames = b[:, : 201 * 320].view(B, 201, 320)
            frame_e = torch.log(wav_frames.pow(2).mean(dim=-1) + 1e-8)
            z_e = (frame_e - frame_e.mean(dim=1, keepdim=True)) / (frame_e.std(dim=1, keepdim=True) + 1e-4)
            w_voiced = F.softmax(2.0 * z_e, dim=1).unsqueeze(-1)
            w_pause = F.softmax(-2.0 * z_e, dim=1).unsqueeze(-1)

            x = b.unsqueeze(1)
            for conv in w2v.feature_extractor.conv_layers:
                x = conv[0](x)
                x = F.layer_norm(x.transpose(-2, -1).float(), (512,), conv[2][1].weight.float(), conv[2][1].bias.float()).transpose(-2, -1).type_as(x)
                x = F.gelu(x)
            x = x.transpose(1, 2)
            x = F.layer_norm(x.float(), (512,), w2v.layer_norm.weight.float(), w2v.layer_norm.bias.float())
            x = w2v.post_extract_proj(x)
            if is_full:
                x = x + w2v.encoder.pos_conv(x.transpose(1, 2)).transpose(1, 2)
                cur = {0: F.layer_norm(x.float(), (1024,))}
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    for l_idx, layer in enumerate(w2v.encoder.layers, start=1):
                        x = layer(x)
                        if l_idx in TAP_12:
                            cur[l_idx] = (
                                F.layer_norm(x.float(), (1024,), w2v.encoder.layer_norm.weight.float(), w2v.encoder.layer_norm.bias.float())
                                if l_idx == 24
                                else F.layer_norm(x.float(), (1024,))
                            )
            else:
                x = x + pos_mod(x.transpose(1, 2)).transpose(1, 2)
                cur = {0: F.layer_norm(x.float(), (1024,))}
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    for l_idx, layer in enumerate(comp_layers, start=1):
                        x = layer(x)
                        if l_idx in TAP_12:
                            cur[l_idx] = F.layer_norm(x.float(), (1024,))
                    x_br = bridge_mod(x)
                    cur[24] = F.layer_norm(x_br.float(), (1024,), w2v.encoder.layer_norm.weight.float(), w2v.encoder.layer_norm.bias.float())

            sl = slice(i, i + B)
            stsp = torch.cat([stft_f, spec_f], dim=1)
            out_rasv5[sl] += stsp @ w_rout_asv5_stsp
            out_rwild[sl] += stsp @ w_rout_wild_stsp
            for p_pos, l_idx in enumerate(TAP_12):
                st_l = torch.nan_to_num(cur[l_idx], 0.0, 0.0, 0.0)
                mu = st_l.mean(dim=1)
                sd_t = st_l.std(dim=1) + 1e-4
                st_cmvn = (st_l - mu.unsqueeze(1)) / sd_t.unsqueeze(1)
                d1 = st_cmvn[:, 1:, :] - st_cmvn[:, :-1, :]
                d2 = d1[:, 1:, :] - d1[:, :-1, :]
                s2 = st_cmvn[:, 2:, :] - st_cmvn[:, :-2, :]
                s4 = st_cmvn[:, 4:, :] - st_cmvn[:, :-4, :]
                d1_norm = d1 / (d1.std(dim=1, keepdim=True) + 1e-4)
                vp_diff = (w_voiced * st_cmvn).sum(dim=1) - (w_pause * st_cmvn).sum(dim=1)
                top_d1 = torch.log(torch.topk(d1.pow(2), k=30, dim=1)[0].mean(dim=1).sqrt() + 1e-4)
                lag1_ch = (st_cmvn * torch.roll(st_cmvn, shifts=1, dims=-1)).mean(dim=1)
                d1_ch = (d1_norm * torch.roll(d1_norm, shifts=1, dims=-1)).mean(dim=1)
                log_s4 = torch.log(s4.std(dim=1) + 1e-4)
                f7_vec = torch.cat([mu.clamp(-20, 20), torch.log(sd_t).clamp(-20, 20), torch.log(d1.std(1) + 1e-4).clamp(-20, 20), torch.log(d2.std(1) + 1e-4).clamp(-20, 20), torch.log(s2.std(1) + 1e-4).clamp(-20, 20), vp_diff.clamp(-20, 20), top_d1.clamp(-20, 20)], dim=1)
                f10_vec = torch.cat([f7_vec, lag1_ch.clamp(-20, 20), d1_ch.clamp(-20, 20), log_s4.clamp(-20, 20)], dim=1)
                z_pos = F.relu((f10_vec - mu_quad_10S[p_pos]) * oinv_quad_10S[p_pos])
                zq = torch.log1p(z_pos * z_pos)

                out_rasv5[sl] += f7_vec @ W_rout_asv5_7S[p_pos]
                out_rwild[sl] += f7_vec @ W_rout_wild_7S[p_pos]
                out_wb_st[sl] += (f10_vec @ W_wb_10S[p_pos]) + (zq @ Wq_wb_10S[p_pos])
                out_wb_wi[sl] += (f10_vec @ W_wb_wild_10S[p_pos]) + (zq @ Wq_wb_wild_10S[p_pos])
                out_pstn_l[sl] += f10_vec @ W_pstn_10S[p_pos]
                out_pstn_q[sl] += zq @ Wq_pstn_10S[p_pos]
                out_asv5_l[sl] += f10_vec @ W_asv5_10S[p_pos]
                out_asv5_q[sl] += zq @ Wq_asv5_10S[p_pos]
                out_inv_l[sl] += f10_vec @ W_inv_10S[p_pos]

            logits, hid = efd.run_aasist_from_ssl(m, cur[24])
            out_ws1[sl] = -logits[:, 1].float()
            hst = torch.cat([hid.float().clamp(-20.0, 20.0), stft_f, spec_f], dim=1)
            qhst = torch.log1p(F.relu((hst - mu_h544) * oinv_h544).pow(2))
            sq_544 = (qhst @ wq_spec544) + bq_spec544
            sh_wb_lin = (hst @ w_spec_wb) + b_spec_wb
            out_sh_ps[sl] = sh_wb_lin
            out_sh_wb[sl] = sh_wb_lin + 0.25 * sq_544
            out_sh_a5[sl] = (hst @ w_spec_asv5) + b_spec_asv5 + 0.15 * sq_544
            out_diffe[sl] = stft_f[:, 33:62].mean(dim=1) - stft_f[:, 3:27].mean(dim=1)

        return torch.stack([out_wb_st, out_wb_wi, out_pstn_l, out_pstn_q, out_asv5_l, out_asv5_q, out_inv_l, out_rasv5, out_rwild, out_ws1, out_sh_wb, out_sh_ps, out_sh_a5, out_diffe], dim=1)

    @torch.inference_mode()
    def predict(wavs1, wavs2=None, wavs3=None, has_c2=None, has_c3=None):
        comps = forward_crop_batch(wavs1)
        if has_c2 is not None and has_c2.any():
            c2_only = has_c2 & (~has_c3)
            if c2_only.any():
                c2_sub_mask = ~has_c3[has_c2]
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
        if is_full:
            bf_mask = torch.sigmoid(2.0 * (1.80 - s_inv_l))
            s_asv5_l_casn = 0.35 * s_asv5_l + 0.65 * s_inv_l - 0.50 * bf_mask * F.relu(s_asv5_l - s_inv_l - 0.50)
            s_asv5_q_casn = (1.0 - 0.45 * bf_mask) * s_asv5_q
            s_pstn_l_casn = 0.90 * s_pstn_l + 0.10 * s_inv_l - 0.35 * bf_mask * F.relu(s_pstn_l - s_inv_l - 0.50)
            g_wild = torch.sigmoid(2.0 * (r_wild - 2.20))
            g_asv5 = torch.sigmoid(4.0 * (r_asv5 - 0.45))
            g_pstn = (1.0 - g_asv5) * torch.sigmoid(12.0 * (-1.52 - diff_e))
            g_wb = torch.clamp(1.0 - g_asv5 - g_pstn, min=0.0)
            sc_pstn = (0.08 * w_s1 + 0.16 * sh_ps + s_pstn_l_casn + s_pstn_q) * 1.3467 - 1.9500
            sc_asv5 = (0.04 * w_s1 + 0.10 * sh_asv5 + s_asv5_l_casn + s_asv5_q_casn) * 1.3500 + 0.8000
            sc_wb = (1.0 - g_wild) * ((0.16 * w_s1 + 0.19 * sh_wb + 0.84 * s_wb_st) * 1.1905) + g_wild * ((0.25 * w_s1 + 0.22 * sh_wb + 0.72 * s_wb_wi) * 1.3889)
            return (g_wb * sc_wb + g_pstn * sc_pstn + g_asv5 * sc_asv5).cpu().numpy().astype(np.float32)

        bf_a5 = torch.sigmoid(2.0 * (4.50 - s_inv_l))
        s_asv5_l_casn = 0.20 * s_asv5_l + 0.80 * s_inv_l - 0.50 * bf_a5 * F.relu(s_asv5_l - s_inv_l - 0.50)
        s_asv5_q_casn = (1.0 - 0.45 * bf_a5) * s_asv5_q
        bf_ps = torch.sigmoid(2.0 * (3.50 - s_inv_l))
        s_pstn_l_casn = 0.90 * s_pstn_l + 0.10 * s_inv_l - 0.35 * bf_ps * F.relu(s_pstn_l - s_inv_l - 0.50)

        g_wild = torch.sigmoid(2.0 * (r_wild - 2.20))
        g_asv5 = torch.sigmoid(4.0 * (r_asv5 - 0.45))
        g_pstn = (1.0 - g_asv5) * torch.sigmoid(12.0 * (-1.52 - diff_e))
        g_wb = torch.clamp(1.0 - g_asv5 - g_pstn, min=0.0)

        sc_pstn_sota = (0.10 * w_s1 + 0.16 * sh_ps + s_pstn_l_casn + s_pstn_q) * 1.3467 - 1.9500
        sc_asv5_sota = (0.04 * w_s1 + 0.10 * sh_asv5 + s_asv5_l_casn + s_asv5_q_casn) * 1.3500 + 0.8000
        sc_wb_hyb = (1.0 - g_wild) * ((0.06 * w_s1 + 0.10 * sh_wb + 0.88 * s_wb_st) * 1.1905) + g_wild * ((0.22 * w_s1 + 0.19 * sh_wb + 0.72 * s_wb_wi) * 1.3889)
        sc_p22h = g_wb * sc_wb_hyb + g_pstn * sc_pstn_sota + g_asv5 * sc_asv5_sota
        return sc_p22h.cpu().numpy().astype(np.float32)

    return predict
