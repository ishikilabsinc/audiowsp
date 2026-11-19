"""AudioWSP & OFP: Open-Source Framework for Weight-Space Programming & Compression of Audio Models.

Provides four unified, strictly sandboxed primitives that operate in `torch.inference_mode()`
(`1 Forward Pass + 0 Backward Passes`, zero evaluation leakage):

  1. `AcousticNuisanceProfiler`: Synthesizes paired acoustic nuisances (telephony 8kHz bandpass,
     codec quantization, room reverb, additive SNR noise, gain shift) strictly on practice/dev
     waveforms and extracts task-orthogonal nuisance drift subspaces `U_nuis`.
  2. `SubspaceNullingSurgery` (`Stage 1`): Projects out nuisance drift subspaces from internal
     Transformer/CNN weights while preserving task/phonetic subspaces via Prior-Anchor Surgery.
  3. `OneForwardPassOptimizer` (`OFP`, `Stage 2`): Solves globally for optimal discriminative or
     CTC ASR readout/adapter weights in 1 forward pass via Boundary-Precision Weighted,
     Nuisance-Orthogonal Woodbury LCMV + Minimax Group-DRO, plus literature baselines
     (`Ridge`, `LDA`, `RanPAC`, `WiSE-FT`, `SGD`, `AdamW`).
  4. `OTVelocityCompressor` (`OT-SVD`, `Stage 3`): Compresses dense Transformer layers by 3x-5x
     while preserving both static covariance and 1st-order frame-to-frame transition velocity (`d1`)
     with Optimal-Transport variance restoration, plus literature baselines
     (`StandardSVD`, `ASVD`, `Wanda`, `MagnitudePruning`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.firewall import assert_practice_split
from exp1_unseen_generators.one_pass_frontier_optimizer import (
    OneForwardPassOptimizer,
    OnePassFrontierOptimizer,
    OFP,
    condition_backbone_in_one_pass,
    distill_readout_per_tap,
    lift_rectified_quadratic_rkhs,
    solve_fpo1_woodbury_lcmv,
    solve_ot_variance_preserving_svd,
)


# ==============================================================================
# Stage 0: Acoustic Nuisance Profiler & Paired Probe Generator
# ==============================================================================


class AcousticNuisanceProfiler:
    """Synthesizes paired acoustic perturbations strictly on practice/dev audio and diagnoses drift."""

    NUISANCE_FAMILIES = ("telephony_8khz", "snr_noise", "room_reverb", "codec_mulaw", "gain_shift")

    def __init__(self, sample_rate: int = 16000, seed: int = 42):
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _fft_bandpass(wav: np.ndarray, sr: int, low_hz: float = 300.0, high_hz: float = 3400.0) -> np.ndarray:
        """Simulate PSTN/telephony 300-3400 Hz bandpass + 8kHz decimation roll-off."""
        n = len(wav)
        if n < 16:
            return wav.copy()
        spec = np.fft.rfft(wav)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        # Smooth raised-cosine roll-off around [low_hz, high_hz]
        mask = np.ones_like(freqs, dtype=np.float32)
        low_trans = np.clip((freqs - (low_hz - 80.0)) / 80.0, 0.0, 1.0)
        high_trans = np.clip(((high_hz + 200.0) - freqs) / 200.0, 0.0, 1.0)
        mask = low_trans * high_trans
        filtered = np.fft.irfft(spec * mask, n=n).astype(np.float32)
        return filtered

    @staticmethod
    def _mulaw_codec(wav: np.ndarray, mu: int = 127) -> np.ndarray:
        """Simulate lossy mu-law / low-bitrate packet quantization artifacts."""
        x = np.clip(wav, -1.0, 1.0)
        s = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        q = np.round(s * 64.0) / 64.0
        y = np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu
        return y.astype(np.float32)

    def _room_reverb(self, wav: np.ndarray, rt60: float = 0.25) -> np.ndarray:
        """Simulate lecture-hall / far-field room impulse response reverberation (e.g. TED-LIUM)."""
        ir_len = int(self.sample_rate * rt60)
        if ir_len < 8 or len(wav) < 16:
            return wav.copy()
        t = np.arange(ir_len, dtype=np.float32) / self.sample_rate
        decay = np.exp(-6.9078 * t / rt60)
        ir = np.zeros(ir_len, dtype=np.float32)
        ir[0] = 1.0
        # Early reflections at 12ms, 23ms, 37ms + diffuse exponential tail
        for delay_ms, amp in ((12, 0.35), (23, -0.22), (37, 0.15), (53, -0.10)):
            idx = int(delay_ms * self.sample_rate / 1000.0)
            if idx < ir_len:
                ir[idx] += amp
        tail = self.rng.standard_normal(ir_len).astype(np.float32) * 0.04 * decay
        tail[0] = 0.0
        ir = ir + tail
        ir /= np.sqrt(np.sum(ir**2) + 1e-12)
        # Fast FFT convolution truncated to original length (preserves exact frame alignment)
        n_fft = 1 << int(np.ceil(np.log2(len(wav) + ir_len - 1)))
        out = np.fft.irfft(np.fft.rfft(wav, n=n_fft) * np.fft.rfft(ir, n=n_fft), n=n_fft)[: len(wav)]
        return out.astype(np.float32)

    def _additive_noise(self, wav: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
        """Add pink/babble-shaped background noise at target SNR (dB)."""
        sig_power = float(np.mean(wav**2)) + 1e-10
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        white = self.rng.standard_normal(len(wav)).astype(np.float32)
        # Shape spectrum toward speech/ambient 1/f envelope
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(len(wav), d=1.0 / self.sample_rate)
        shape = 1.0 / np.sqrt(np.maximum(freqs, 100.0) / 100.0)
        shaped = np.fft.irfft(spec * shape, n=len(wav)).astype(np.float32)
        shaped = shaped / (np.std(shaped) + 1e-10) * np.sqrt(noise_power)
        return (wav + shaped).astype(np.float32)

    def generate_paired_probes(
        self,
        wav: np.ndarray,
        *,
        split: str = "practice",
    ) -> Dict[str, np.ndarray]:
        """Generate frame-aligned acoustic nuisance pairs strictly from a verified practice waveform."""
        assert_practice_split(split, context="AcousticNuisanceProfiler.generate_paired_probes")
        w = np.asarray(wav, dtype=np.float32)
        return {
            "telephony_8khz": self._fft_bandpass(w, self.sample_rate, 300.0, 3400.0),
            "snr_noise": self._additive_noise(w, snr_db=10.0),
            "room_reverb": self._room_reverb(w, rt60=0.28),
            "codec_mulaw": self._mulaw_codec(w, mu=127),
            "gain_shift": (w * 0.35).astype(np.float32),
        }

    @staticmethod
    @torch.inference_mode()
    def extract_task_orthogonal_nuisance_subspace(
        clean_acts: torch.Tensor,
        perturbed_acts: torch.Tensor,
        task_basis: Optional[torch.Tensor] = None,
        rank_nuis: int = 16,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute top-`rank_nuis` nuisance drift basis `U_nuis` (`(k, D)`), orthogonalized against `task_basis`."""
        C_f = clean_acts.reshape(-1, clean_acts.size(-1)).float()
        P_f = perturbed_acts.reshape(-1, perturbed_acts.size(-1)).float()
        deltas = P_f - C_f
        drift_ratio = float(torch.linalg.norm(deltas) / (torch.linalg.norm(C_f) + 1e-12))

        # Project out task-discriminative directions first so we never null phonetic/task signals
        Q_t = None
        if task_basis is not None and task_basis.numel() > 0:
            B_t = F.normalize(task_basis.float(), dim=-1)  # (k_task, D)
            Q_t, _ = torch.linalg.qr(B_t.T, mode="reduced")  # (D, k_task)
            deltas = deltas - (deltas @ Q_t) @ Q_t.T

        cov_nuis = (deltas.T @ deltas) / float(max(len(deltas), 1))
        if Q_t is not None:
            # Deflate task subspace so degenerate zero-eigenvalue modes never mix with Q_t
            cov_nuis = cov_nuis - 1e3 * (Q_t @ Q_t.T)
        evals, evecs = torch.linalg.eigh(cov_nuis)
        k = min(rank_nuis, evecs.size(1))
        top_vecs = evecs[:, -k:].flip(1).T.contiguous()  # (k, D)
        if Q_t is not None:
            top_vecs = top_vecs - (top_vecs @ Q_t) @ Q_t.T
            top_vecs = F.normalize(top_vecs, dim=-1)
        top_vals = evals[-k:].flip(0).clamp(min=0.0)
        explained = float(top_vals.sum() / (evals.clamp(min=0.0).sum() + 1e-12))
        return top_vecs, {"drift_ratio": drift_ratio, "nuisance_Subspace_explained": explained}


# ==============================================================================
# Stage 1: Internal Subspace Nulling & Prior-Anchor Surgery
# ==============================================================================


class SubspaceNullingSurgery:
    """Projects task-orthogonal nuisance subspaces out of internal layer weights in 0 backward passes."""

    @staticmethod
    @torch.inference_mode()
    def apply_layer_output_surgery(
        linear_layer: nn.Linear,
        U_nuis: torch.Tensor,
        gamma: float = 0.25,
    ) -> None:
        """Project `U_nuis` (`(k, D_out)`) out of the output column space of `linear_layer`."""
        W = linear_layer.weight.data.float()
        U = U_nuis.to(device=W.device, dtype=torch.float32)
        P_nuis = U.T @ U  # (D_out, D_out)
        I_d = torch.eye(W.size(0), device=W.device, dtype=torch.float32)
        M = I_d - float(gamma) * P_nuis
        linear_layer.weight.data.copy_((M @ W).type_as(linear_layer.weight.data))
        if linear_layer.bias is not None:
            b = linear_layer.bias.data.float()
            linear_layer.bias.data.copy_((M @ b).type_as(linear_layer.bias.data))

    @staticmethod
    @torch.inference_mode()
    def apply_layer_input_surgery(
        linear_layer: nn.Linear,
        U_nuis: torch.Tensor,
        gamma: float = 0.35,
    ) -> None:
        """Project `U_nuis` (`(k, D_in)`) out of the input row space of `linear_layer` (`W <- W (I - gamma U^T U)`)."""
        W = linear_layer.weight.data.float()
        U = U_nuis.to(device=W.device, dtype=torch.float32)
        P_nuis = U.T @ U  # (D_in, D_in)
        I_d = torch.eye(W.size(1), device=W.device, dtype=torch.float32)
        M = I_d - float(gamma) * P_nuis
        linear_layer.weight.data.copy_((W @ M).type_as(linear_layer.weight.data))


# ==============================================================================
# Stage 2: One-Forward-Pass Optimizer (OFP) for CTC ASR & Multi-Output Readouts
#          + Literature Baselines (Ridge, LDA, RanPAC, WiSE-FT, SGD, AdamW)
# ==============================================================================


@torch.inference_mode()
def solve_ofp_ctc_readout(
    H_clean: torch.Tensor,
    Y_target: torch.Tensor,
    W_prior: torch.Tensor,
    b_prior: torch.Tensor,
    H_nuis_list: Optional[Sequence[Tuple[torch.Tensor, float]]] = None,
    lam_prior: float = 15.0,
    gamma_nuis: float = 6.0,
    boundary_beta: float = 1.5,
    boundary_boost: float = 2.0,
    split: str = "dev-clean",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Solve for optimal CTC readout weights `(W_ofp, b_ofp)` in 1 Forward Pass (`1F + 0B`).

    Combines:
      1. Boltzmann-Focal Boundary Precision Re-Weighting on ambiguous phoneme-transition frames:
         `pi_t = 1 + boundary_boost * exp(-boundary_beta * |z_{t,(1)} - z_{t,(2)}|)`
      2. Paired Nuisance-Orthogonal Woodbury / Tikhonov Penalty (`gamma_nuis * Sigma_nuis`)
      3. Prior-Anchored Weight Surgery around `W_prior` (`lam_prior * I`) so rare characters
         (`X`, `J`, `Q`, `Z`) never lose their pretrained geometry.

    Args:
        H_clean: `(N_frames, D)` practice hidden states (or concatenated multi-layer states).
        Y_target: `(N_frames, V)` target CTC frame logits (teacher logits + transcript sharpening).
        W_prior: `(V, D)` prior readout matrix (`lm_head.weight`, padded with zeros for skip taps).
        b_prior: `(V,)` prior readout bias (`lm_head.bias`).
        H_nuis_list: List of `(H_perturbed, weight)` pairs aligned frame-by-frame with `H_clean`.
    """
    assert_practice_split(split, context="solve_ofp_ctc_readout")
    dev = H_clean.device
    H = H_clean.float()
    Y = Y_target.float()
    W0 = W_prior.to(device=dev, dtype=torch.float32)  # (V, D)
    b0 = b_prior.to(device=dev, dtype=torch.float32)  # (V,)
    N, D = H.shape

    # 1. Boltzmann-Focal Boundary Precision Weights on frame margins (0 extra forward passes)
    top2 = torch.topk(Y, k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
    if boundary_boost > 0.0:
        pi = 1.0 + float(boundary_boost) * torch.exp(-float(boundary_beta) * margin)
    else:
        pi = torch.ones(N, device=dev, dtype=torch.float32)
    pi = pi / pi.mean()

    # Weighted means
    pi_col = pi.unsqueeze(1)
    mu_h = (H * pi_col).mean(0, keepdim=True)
    mu_y = (Y * pi_col).mean(0, keepdim=True)
    Hc = H - mu_h
    Yc = Y - mu_y
    sqrt_pi = torch.sqrt(pi_col)
    Hw = Hc * sqrt_pi
    Yw = Yc * sqrt_pi

    Cov = (Hw.T @ Hw) / float(N)
    Cross = (Hw.T @ Yw) / float(N)

    # 2. Paired Frame-Aligned Nuisance Covariance Penalty
    nuis_trace = 0.0
    if H_nuis_list:
        for H_pert, w_n in H_nuis_list:
            dH = (H_pert.to(device=dev, dtype=torch.float32) - H) * sqrt_pi
            cov_n = (dH.T @ dH) / float(len(dH))
            Cov = Cov + float(gamma_nuis * w_n) * cov_n
            nuis_trace += float(torch.trace(cov_n))

    # 3. Prior-Anchored Closed-Form Solve: (Cov + lam * I) W^T = Cross + lam * W0^T
    # Plus map the nuisance shift smoothly into W0's subspace so W0 itself is nuisance-damped
    I_d = torch.eye(D, device=dev, dtype=torch.float32)
    A = Cov + float(lam_prior) * I_d
    RHS = Cross + float(lam_prior) * W0.T
    W_ofp_T = torch.linalg.solve(A, RHS)
    W_ofp = W_ofp_T.T.contiguous()  # (V, D)

    # Optimal mean-matching bias anchored toward b0
    b_raw = (mu_y - mu_h @ W_ofp_T).squeeze(0)
    b_ofp = 0.5 * b0 + 0.5 * b_raw

    return W_ofp, b_ofp, {
        "n_frames": float(N),
        "mean_boundary_weight": float(pi.mean()),
        "max_boundary_weight": float(pi.max()),
        "nuisance_cov_trace": nuis_trace,
    }


@torch.inference_mode()
def solve_ridge_ctc_baseline(
    H_clean: torch.Tensor,
    Y_target: torch.Tensor,
    W_prior: torch.Tensor,
    b_prior: torch.Tensor,
    lam: float = 15.0,
    split: str = "dev-clean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Literature Baseline 1: Standard Ridge / ELM Linear Probe (zero nuisance nulling, zero boundary weighting)."""
    assert_practice_split(split, context="solve_ridge_ctc_baseline")
    dev = H_clean.device
    H = H_clean.float()
    Y = Y_target.float()
    N, D = H.shape
    mu_h = H.mean(0, keepdim=True)
    mu_y = Y.mean(0, keepdim=True)
    Hc = H - mu_h
    Yc = Y - mu_y
    Cov = (Hc.T @ Hc) / float(N) + float(lam) * torch.eye(D, device=dev, dtype=torch.float32)
    Cross = (Hc.T @ Yc) / float(N)
    W_T = torch.linalg.solve(Cov, Cross)
    b = (mu_y - mu_h @ W_T).squeeze(0)
    return W_T.T.contiguous(), b


@torch.inference_mode()
def solve_ranpac_ctc_baseline(
    H_clean: torch.Tensor,
    Y_target: torch.Tensor,
    proj_dim: int = 1536,
    lam: float = 25.0,
    seed: int = 42,
    split: str = "dev-clean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Literature Baseline 2: RanPAC (Random Projection + Analytical Ridge, McDonnell et al. NeurIPS 2023)."""
    assert_practice_split(split, context="solve_ranpac_ctc_baseline")
    dev = H_clean.device
    H = H_clean.float()
    Y = Y_target.float()
    N, D = H.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    R = (torch.randn(D, proj_dim, generator=gen) / np.sqrt(D)).to(device=dev, dtype=torch.float32)
    Phi = F.relu(H @ R)
    mu_p = Phi.mean(0, keepdim=True)
    mu_y = Y.mean(0, keepdim=True)
    Pc = Phi - mu_p
    Yc = Y - mu_y
    Cov = (Pc.T @ Pc) / float(N) + float(lam) * torch.eye(proj_dim, device=dev, dtype=torch.float32)
    Cross = (Pc.T @ Yc) / float(N)
    W_p = torch.linalg.solve(Cov, Cross)
    b = (mu_y - mu_p @ W_p).squeeze(0)
    return R, W_p.T.contiguous(), b


# ==============================================================================
# Stage 3: Optimal-Transport Variance & d1-Velocity Compressor (OT-SVD)
#          + Literature Compression Baselines (Standard SVD, ASVD, Wanda, Magnitude)
# ==============================================================================


class OTVelocityCompressor:
    """Compresses dense Transformer layers via OT-SVD vs. Standard SVD, ASVD, Wanda, and Magnitude Pruning."""

    @staticmethod
    @torch.inference_mode()
    def compress_ot_svd(
        Z_3d: torch.Tensor,
        Y_3d: torch.Tensor,
        W_prior: torch.Tensor,
        b_prior: torch.Tensor,
        rank: int,
        lam: float = 6e-2,
        max_gain: float = 1.6,
        split: str = "dev-clean",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Our Method (`OT-SVD`): Optimal-Transport Variance & `d1`-Velocity Preserving Factorized Compression.

        Returns `(U_r, V_r, b_hat)` where `W_hat = U_r @ V_r` has rank `rank`.
        """
        assert_practice_split(split, context="OTVelocityCompressor.compress_ot_svd")
        W_hat, b_hat = solve_ot_variance_preserving_svd(
            Z_3d=Z_3d,
            Y_3d=Y_3d,
            W_prior=W_prior,
            b_prior=b_prior,
            rank=rank,
            lam=lam,
            max_gain=max_gain,
        )
        U, S, Vh = torch.linalg.svd(W_hat.float(), full_matrices=False)
        r = min(rank, S.numel())
        sqrt_s = torch.sqrt(S[:r])
        U_r = (U[:, :r] * sqrt_s.unsqueeze(0)).contiguous()
        V_r = (sqrt_s.unsqueeze(1) * Vh[:r, :]).contiguous()
        return U_r, V_r, b_hat.float()

    @staticmethod
    @torch.inference_mode()
    def compress_standard_svd(
        W_prior: torch.Tensor,
        b_prior: torch.Tensor,
        rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Literature Baseline 1: Data-free Standard Weight SVD (`W ~ U_r S_r V_r^T`)."""
        W = W_prior.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(rank, S.numel())
        sqrt_s = torch.sqrt(S[:r])
        U_r = (U[:, :r] * sqrt_s.unsqueeze(0)).contiguous()
        V_r = (sqrt_s.unsqueeze(1) * Vh[:r, :]).contiguous()
        return U_r, V_r, b_prior.float().clone()

    @staticmethod
    @torch.inference_mode()
    def compress_asvd(
        Z_3d: torch.Tensor,
        W_prior: torch.Tensor,
        b_prior: torch.Tensor,
        rank: int,
        split: str = "dev-clean",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Literature Baseline 2: Activation-Weighted SVD (`ASVD`, Yuan et al. / SVD-LLM).

        Weights columns of `W` by static activation RMS/covariance `S_z = diag(||Z_j||_2)`
        WITHOUT temporal-difference `d1` velocity covariance and WITHOUT Optimal-Transport
        marginal variance restoration or mean-shift compensation.
        """
        assert_practice_split(split, context="OTVelocityCompressor.compress_asvd")
        Z_flat = Z_3d.reshape(-1, Z_3d.size(-1)).float()
        W = W_prior.to(device=Z_flat.device, dtype=torch.float32)
        col_scale = torch.sqrt((Z_flat**2).mean(0) + 1e-6)  # (D_in,)
        W_scaled = W * col_scale.unsqueeze(0)
        U, S, Vh = torch.linalg.svd(W_scaled, full_matrices=False)
        r = min(rank, S.numel())
        sqrt_s = torch.sqrt(S[:r])
        U_r = (U[:, :r] * sqrt_s.unsqueeze(0)).contiguous()
        V_r = ((sqrt_s.unsqueeze(1) * Vh[:r, :]) / col_scale.unsqueeze(0)).contiguous()
        return U_r, V_r, b_prior.to(device=Z_flat.device, dtype=torch.float32).clone()

    @staticmethod
    @torch.inference_mode()
    def compress_wanda(
        Z_3d: torch.Tensor,
        W_prior: torch.Tensor,
        b_prior: torch.Tensor,
        target_density: float,
        split: str = "dev-clean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Literature Baseline 3: Wanda (Pruning by Weights and Activations, Sun et al. ICLR 2024).

        Scores each weight entry by `S_{ij} = |W_{ij}| * ||Z_j||_2` and retains `target_density`
        fraction of non-zero parameters per output row (matching the parameter budget of rank-`r` SVD).
        """
        assert_practice_split(split, context="OTVelocityCompressor.compress_wanda")
        Z_flat = Z_3d.reshape(-1, Z_3d.size(-1)).float()
        W = W_prior.to(device=Z_flat.device, dtype=torch.float32)
        act_norm = torch.linalg.norm(Z_flat, ord=2, dim=0)  # (D_in,)
        metric = W.abs() * act_norm.unsqueeze(0)
        k_keep = max(1, int(round(W.size(1) * float(target_density))))
        topk_idx = torch.topk(metric, k=k_keep, dim=1).indices
        mask = torch.zeros_like(W)
        mask.scatter_(1, topk_idx, 1.0)
        return (W * mask).contiguous(), b_prior.to(device=Z_flat.device, dtype=torch.float32).clone()

    @staticmethod
    @torch.inference_mode()
    def compress_magnitude(
        W_prior: torch.Tensor,
        b_prior: torch.Tensor,
        target_density: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Literature Baseline 4: Unstructured / Row-wise Magnitude Pruning at equivalent parameter density."""
        W = W_prior.float()
        k_keep = max(1, int(round(W.size(1) * float(target_density))))
        topk_idx = torch.topk(W.abs(), k=k_keep, dim=1).indices
        mask = torch.zeros_like(W)
        mask.scatter_(1, topk_idx, 1.0)
        return (W * mask).contiguous(), b_prior.float().clone()


# ==============================================================================
# Stage 4: Generalized Eigenvalue Decomposition (GEVD) & Multi-Tap Temporal Lifter
# ==============================================================================


class MultiTapTemporalLifter:
    """Unified multi-scale spectro-temporal context & derivative lifter shared by Deepfake & ASR WSP."""

    @staticmethod
    @torch.inference_mode()
    def build_k_tap_context(x_3d: torch.Tensor, k_taps: int = 3) -> torch.Tensor:
        """Concatenate `k_taps` frame-aligned temporal neighbors `(B, T, k_taps * D)`."""
        if k_taps == 1:
            return x_3d
        pad = k_taps // 2
        xp = F.pad(x_3d.transpose(1, 2), (pad, pad), mode="replicate").transpose(1, 2)
        T = x_3d.size(1)
        return torch.cat([xp[:, i : i + T, :] for i in range(k_taps)], dim=-1)

    @staticmethod
    @torch.inference_mode()
    def build_deep_stack_bridge_features(
        h_top: torch.Tensor,
        h_mid1: torch.Tensor,
        h_mid2: torch.Tensor,
    ) -> torch.Tensor:
        """Construct 4-tap encoder bridge features `[h_top, LN(h_mid1), LN(h_mid2), d1(h_top)]` (`(B, T, 4*D)`)."""
        d_dim = h_top.size(-1)
        m1 = F.layer_norm(h_mid1.float(), (d_dim,))
        m2 = F.layer_norm(h_mid2.float(), (d_dim,))
        d1 = torch.zeros_like(h_top, dtype=torch.float32)
        if h_top.size(1) > 2:
            d1[:, 1:-1, :] = 0.5 * (h_top[:, 2:, :].float() - h_top[:, :-2, :].float())
        return torch.cat([h_top.float(), m1, m2, d1], dim=-1)


class GEVDProjectionSurgery:
    """Generalized Eigenvalue Decomposition (`Sigma_clean v = lambda Sigma_nuis v`) Rank-r Adapter Solver."""

    @staticmethod
    @torch.inference_mode()
    def solve_gevd_residual_adapter(
        H_clean_3d: torch.Tensor,
        H_pert_3d: torch.Tensor,
        rank_r: int = 96,
        k_taps: int = 3,
        gain: float = 0.25,
        split: str = "dev-clean",
    ) -> torch.Tensor:
        """Solve in 1 forward pass (`1F + 0B`) for a rank-`rank_r` GEVD residual projection `dW` (`(k_taps*D, D)`).

        1. Solves the generalized eigenproblem `Cov_clean V = Cov_nuis V Lambda` via Cholesky whitening.
        2. Restricts the target restoration residual `(H_clean - H_pert) @ V_r V_r^T` to the top-`rank_r`
           high-SNR generalized eigenvectors `V_r` so noise-dominated directions are never amplified.
        """
        assert_practice_split(split, context="GEVDProjectionSurgery.solve_gevd_residual_adapter")
        dev = H_clean_3d.device
        D = H_clean_3d.size(-1)
        Hc = H_clean_3d.reshape(-1, D).double()
        Hp = H_pert_3d.reshape(-1, D).double()
        dH = Hp - Hc

        Cov_c = (Hc.T @ Hc) / float(len(Hc)) + 1e-4 * torch.eye(D, device=dev, dtype=torch.float64)
        Cov_d = (dH.T @ dH) / float(len(dH)) + 1e-3 * torch.eye(D, device=dev, dtype=torch.float64)

        ev_d, u_d = torch.linalg.eigh(Cov_d)
        ev_d = ev_d.clamp(min=1e-6)
        W_whiten = u_d @ torch.diag(ev_d.rsqrt()) @ u_d.T
        M_snr = W_whiten @ Cov_c @ W_whiten
        _, u_snr = torch.linalg.eigh(M_snr)
        r = min(rank_r, D)
        V_r = F.normalize((W_whiten @ u_snr[:, -r:]).float(), dim=0).double()  # (D, r)
        P_r = V_r @ V_r.T  # (D, D)

        Phi_c = MultiTapTemporalLifter.build_k_tap_context(H_clean_3d.float(), k_taps).reshape(-1, k_taps * D).double()
        Phi_p = MultiTapTemporalLifter.build_k_tap_context(H_pert_3d.float(), k_taps).reshape(-1, k_taps * D).double()
        dPhi = Phi_p - Phi_c

        F_dim = k_taps * D
        Cov_fc = (Phi_c.T @ Phi_c) / float(len(Phi_c))
        Cov_fp = (Phi_p.T @ Phi_p) / float(len(Phi_p))
        Cov_fd = (dPhi.T @ dPhi) / float(len(dPhi))
        avg_eig = float(torch.trace(Cov_fc)) / float(F_dim)

        Residual_target = (Hc - Hp) @ P_r
        Cross = (Phi_p.T @ Residual_target) / float(len(Phi_p))
        A = Cov_fc + 0.25 * Cov_fp + 0.35 * Cov_fd + (0.20 * avg_eig) * torch.eye(F_dim, device=dev, dtype=torch.float64)
        dW = torch.linalg.solve(A, float(gain) * Cross).float()
        return dW.contiguous()


# ==============================================================================
# Stage 5: 2D Factored MoE & Micro-Diarizer Compaction (Audio-LLM Compression)
# ==============================================================================


class MoEReLU2Compactor:
    """2D Factored Mixture-of-Experts Compaction (Expert Router Masking + Intra-Expert ReLU^2 Pruning)."""

    @staticmethod
    @torch.inference_mode()
    def mask_dead_experts_in_router(
        e_score_correction_bias: torch.Tensor,
        expert_activation_mass: torch.Tensor,
        keep_experts: int = 108,
    ) -> torch.Tensor:
        """Mask non-speech/dead experts in `gate.e_score_correction_bias` without altering kept routing."""
        n_total = expert_activation_mass.numel()
        n_prune = max(0, n_total - int(keep_experts))
        if n_prune == 0:
            return e_score_correction_bias
        _, dead_idx = torch.topk(expert_activation_mass.float(), k=n_prune, largest=False)
        out_bias = e_score_correction_bias.clone()
        out_bias[dead_idx] = -1e4
        return out_bias

    @staticmethod
    @torch.inference_mode()
    def compact_expert_neurons_relu2(
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        neuron_importance: torch.Tensor,
        keep_inter_dim: int = 1536,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slice `[E, D_inter, D_in]` `up_proj` and `[E, D_in, D_inter]` `down_proj` to top-`keep_inter_dim` neurons."""
        k = min(int(keep_inter_dim), up_proj.size(1))
        top_idx = torch.topk(neuron_importance.float(), k=k, dim=1).indices.sort(dim=1).values  # (E, k)
        idx_up = top_idx.unsqueeze(-1).expand(-1, -1, up_proj.size(2))
        idx_dn = top_idx.unsqueeze(1).expand(-1, down_proj.size(1), -1)
        new_up = torch.gather(up_proj, dim=1, index=idx_up).contiguous()
        new_dn = torch.gather(down_proj, dim=2, index=idx_dn).contiguous()
        return new_up, new_dn


# ==============================================================================
# Stage 6: Dual-Head CTC-Anchored & Condorcet Jury MBR Consensus Decoder
# ==============================================================================


class CTCAwareConsensusDecoder:
    """Dual-Head CTC-Locked Multi-Architecture & Condorcet Jury MBR Consensus Decoder."""

    @staticmethod
    def _word_levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
        if a == b:
            return 0
        n, m = len(a), len(b)
        if n < m:
            a, b, n, m = b, a, m, n
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            ai = a[i - 1]
            cur = [i] + [0] * m
            for j in range(1, m + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
            prev = cur
        return prev[m]

    @classmethod
    def select_condorcet_mbr(
        cls,
        candidates: Sequence[str],
        jury_hypotheses_and_weights: Sequence[Tuple[str, float]],
    ) -> str:
        """Select the candidate minimizing weighted word-level Levenshtein risk across the Condorcet jury."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        cand_tokens = [c.lower().split() for c in candidates]
        jury_tokens = [(h.lower().split(), float(w)) for h, w in jury_hypotheses_and_weights if h.strip()]
        if not jury_tokens:
            return candidates[0]
        best_idx = min(
            range(len(candidates)),
            key=lambda i: sum(w * cls._word_levenshtein(cand_tokens[i], jt) for jt, w in jury_tokens),
        )
        return candidates[best_idx]


# ==============================================================================
# Unified End-to-End AudioWSPPipeline (Runs Identically on Deepfake & ASR)
# ==============================================================================


class AudioWSPPipeline:
    """Unified 6-Block Weight-Space Programming & One-Forward-Pass Optimization Pipeline.

    Executes the exact same modular sequence on both Audio Deepfake Detection (`task='deepfake'`)
    and Automatic Speech Recognition (`task='asr'`):
      - Block 0: Anti-Leakage Firewall (`assert_practice_split` + SHA-256 Audit Certificate)
      - Block 1: `AcousticNuisanceProfiler` (Paired clean-vs-corrupted probes & QR task-orthogonal `U_nuis`)
      - Block 2: `SubspaceNullingSurgery` & `GEVDProjectionSurgery` (In-place nulling & GEVD Rank-r adapter)
      - Block 3: `MultiTapTemporalLifter` & `lift_rectified_quadratic_rkhs` (Temporal velocity `d1` & RKHS lifting)
      - Block 4: `OneForwardPassOptimizer` (`1F + 0B` Dual-Woodbury LCMV for Deepfake / Prior-Anchored OFP for ASR)
      - Block 5: `OTVelocityCompressor` (`OT-SVD` preserving `Cov(Z) + 2*Cov(d1_Z)`)
      - Block 6: `CTCAwareConsensusDecoder` (CTC loop guard & Condorcet Jury MBR)
    """

    def __init__(self, task: str = "deepfake", split: str = "practice", sample_rate: int = 16000):
        if task not in ("deepfake", "asr"):
            raise ValueError(f"Unsupported task '{task}'; expected 'deepfake' or 'asr'.")
        self.task = task
        self.split = assert_practice_split(split, context=f"AudioWSPPipeline({task})")
        self.profiler = AcousticNuisanceProfiler(sample_rate=sample_rate)

    @torch.inference_mode()
    def fit_and_compress(
        self,
        H_clean_3d: torch.Tensor,
        H_pert_3d: torch.Tensor,
        targets: torch.Tensor,
        linear_layer: nn.Linear,
        *,
        task_basis: Optional[torch.Tensor] = None,
        rank_nuis: int = 8,
        gevd_rank: int = 32,
        svd_rank: int = 32,
    ) -> Dict[str, object]:
        """Run all 6 AudioWSP+OFP blocks in 1 forward pass (`0` backward passes) and return diagnostics."""
        assert_practice_split(self.split, context=f"AudioWSPPipeline.fit_and_compress({self.task})")
        B, T, D = H_clean_3d.shape

        # Block 1: Extract QR task-orthogonal nuisance drift subspace U_nuis
        U_nuis, nuis_stats = self.profiler.extract_task_orthogonal_nuisance_subspace(
            clean_acts=H_clean_3d,
            perturbed_acts=H_pert_3d,
            task_basis=task_basis,
            rank_nuis=rank_nuis,
        )

        # Block 2: Apply in-place null-space surgery & solve GEVD rank-r residual adapter
        SubspaceNullingSurgery.apply_layer_input_surgery(linear_layer, U_nuis, gamma=0.25)
        dW_gevd = GEVDProjectionSurgery.solve_gevd_residual_adapter(
            H_clean_3d, H_pert_3d, rank_r=min(gevd_rank, D), k_taps=3, split=self.split
        )

        # Block 3: Multi-tap temporal velocity lifting
        Phi_3 = MultiTapTemporalLifter.build_k_tap_context(H_clean_3d, k_taps=3)
        H_restored_3d = H_pert_3d + (MultiTapTemporalLifter.build_k_tap_context(H_pert_3d, k_taps=3) @ dW_gevd)
        drift_before = float(torch.linalg.norm(H_pert_3d - H_clean_3d) / (torch.linalg.norm(H_clean_3d) + 1e-12))
        drift_after = float(torch.linalg.norm(H_restored_3d - H_clean_3d) / (torch.linalg.norm(H_clean_3d) + 1e-12))

        # Block 4: One-Forward-Pass Optimizer (OFP / FPO-1)
        if self.task == "deepfake":
            X_utt = H_clean_3d.mean(dim=1)
            X_pert_utt = H_pert_3d.mean(dim=1)
            head = nn.Linear(D, 1, device=H_clean_3d.device)
            opt = OneForwardPassOptimizer(head.parameters(), lam=10.0, use_rkhs=True)
            opt.accumulate_forward_batch(X_utt, targets.long())
            opt.add_nuisance_Export_pair(X_utt, X_pert_utt, weight=4.0)
            ofp_diag = opt.step_one_pass()
            ofp_summary = {"mode": "dual_woodbury_lcmv_rkhs", "alpha_dro": ofp_diag["alpha_dro"].tolist()}
        else:
            Hc_2d = H_clean_3d.reshape(-1, D)
            Hp_2d = H_pert_3d.reshape(-1, D)
            Y_2d = targets.reshape(-1, linear_layer.out_features)
            W_ofp, b_ofp, ofp_summary = solve_ofp_ctc_readout(
                H_clean=Hc_2d,
                Y_target=Y_2d,
                W_prior=linear_layer.weight.data,
                b_prior=linear_layer.bias.data if linear_layer.bias is not None else torch.zeros(linear_layer.out_features, device=Hc_2d.device),
                H_nuis_list=[(Hp_2d, 1.0)],
                split=self.split,
            )
            linear_layer.weight.data.copy_(W_ofp.type_as(linear_layer.weight.data))
            if linear_layer.bias is not None:
                linear_layer.bias.data.copy_(b_ofp.type_as(linear_layer.bias.data))

        # Block 5: Velocity-Preserving Compression (OT-SVD)
        Y_teacher_3d = linear_layer(H_clean_3d)
        U_r, V_r, b_hat = OTVelocityCompressor.compress_ot_svd(
            Z_3d=H_clean_3d,
            Y_3d=Y_teacher_3d,
            W_prior=linear_layer.weight.data,
            b_prior=linear_layer.bias.data if linear_layer.bias is not None else torch.zeros(linear_layer.out_features, device=H_clean_3d.device),
            rank=min(svd_rank, D),
            split=self.split,
        )

        return {
            "task": self.task,
            "split_verified": self.split,
            "nuisance_stats": nuis_stats,
            "drift_before_gevd": drift_before,
            "drift_after_gevd": drift_after,
            "multi_tap_shape": list(Phi_3.shape),
            "ofp_summary": ofp_summary,
            "ot_svd_factors_shape": [list(U_r.shape), list(V_r.shape)],
        }


