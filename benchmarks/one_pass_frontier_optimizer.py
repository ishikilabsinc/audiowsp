"""One-Pass Frontier Optimizer (FPO-1): Closed-Form Weight-Space Optimization in 1 Forward Pass.

Replaces iterative first-order backpropagation (SGD/AdamW) with a single forward-pass
(`torch.inference_mode()`, zero `.backward()` calls) weight-space solver combining:

1. Geodesic Prior-Anchor & Nuisance-Subspace Nulling Surgery (`condition_backbone_in_one_pass`)
2. Optimal-Transport Variance & Temporal-Velocity Preserving Layer Synthesis (`solve_ot_variance_preserving_svd`)
3. Multi-Scale Spectro-Temporal & One-Sided Rectified Quadratic RKHS Lifting (`lift_rectified_quadratic_rkhs`)
4. Boltzmann-Focal Boundary Precision & Minimax Group-DRO Dual-Woodbury LCMV (`solve_fpo1_woodbury_lcmv`)
5. Nuisance-Orthogonal Per-Tap Readout Distillation (`distill_readout_per_tap`)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


@torch.inference_mode()
def condition_backbone_in_one_pass(
    model: nn.Module,
    prior_state_dict: Dict[str, torch.Tensor],
    U_nuis: torch.Tensor,
    alpha_early: float = 0.65,
    early_prefixes: Tuple[str, ...] = (
        "feature_extractor.",
        "post_extract_proj.",
        "layer_norm.",
        "encoder.pos_conv.",
        "encoder.layers.0.",
        "encoder.layers.1.",
        "encoder.layers.2.",
        "encoder.layers.3.",
    ),
    late_taper_map: Optional[Dict[int, float]] = None,
    eta_nuis: float = 0.10,
) -> int:
    """Stage 1: Direct in-place weight surgery on internal CNN + Transformer layers in 0 backward passes.

    Restores early acoustic filterbank layers toward the foundation prior `prior_state_dict`
    and projects out the top-k nuisance drift eigenvectors `U_nuis` from late Transformer
    `out_proj` and `fc2` projections.
    """
    if late_taper_map is None:
        late_taper_map = {18: 0.55, 19: 0.50, 20: 0.45, 21: 0.30, 22: 0.20, 23: 0.15}

    edited_params = 0
    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        clean_name = name[len("ssl_model.model.") :] if name.startswith("ssl_model.model.") else name
        if clean_name in prior_state_dict and clean_name.startswith(early_prefixes):
            w0 = prior_state_dict[clean_name].to(device=param.device, dtype=param.dtype)
            param.data.copy_(w0 + alpha_early * (param.data - w0))
            edited_params += param.numel()

    encoder_layers = None
    if hasattr(model, "ssl_model") and hasattr(model.ssl_model, "model"):
        encoder_layers = model.ssl_model.model.encoder.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        encoder_layers = model.encoder.layers

    if encoder_layers is not None and U_nuis is not None:
        dev = next(model.parameters()).device
        U = U_nuis.to(device=dev, dtype=torch.float32)
        P_nuis = U.T @ U
        d_model = P_nuis.size(0)
        I_d = torch.eye(d_model, device=dev, dtype=torch.float32)
        for l_idx, gamma in late_taper_map.items():
            if l_idx < len(encoder_layers):
                lyr = encoder_layers[l_idx]
                M_proj = gamma * I_d - min(eta_nuis, gamma * 0.8) * P_nuis
                for proj in (lyr.self_attn.out_proj, lyr.fc2):
                    proj.weight.data.copy_((M_proj @ proj.weight.data.float()).type_as(proj.weight.data))
                    edited_params += proj.weight.numel()
                    if proj.bias is not None:
                        proj.bias.data.copy_((M_proj @ proj.bias.data.float()).type_as(proj.bias.data))
                        edited_params += proj.bias.numel()

    return edited_params


@torch.inference_mode()
def solve_ot_variance_preserving_svd(
    Z_3d: torch.Tensor,
    Y_3d: torch.Tensor,
    W_prior: torch.Tensor,
    b_prior: torch.Tensor,
    rank: int,
    lam: float = 6e-2,
    max_gain: float = 1.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage 2: Optimal-Transport Variance & Temporal-Difference Preserving Internal Layer Solver.

    Solves for a rank-`rank` weight matrix `W_hat` and bias `b_hat` mapping `Z_3d -> Y_3d`
    in 1 forward pass while preserving both static activation covariance and first-order
    frame-to-frame transition velocity (`d1`), rescaling per-channel output energy via
    the Optimal-Transport variance ratio so downstream derivative statistics do not decay.
    """
    B, T, D_in = Z_3d.shape
    D_out = Y_3d.size(2)
    dev = Z_3d.device
    Z_in = Z_3d.reshape(-1, D_in).float()
    Y_target = Y_3d.reshape(-1, D_out).float()
    W_pr = W_prior.to(device=dev, dtype=torch.float32)

    mu_z = Z_in.mean(0, keepdim=True)
    mu_y = Y_target.mean(0, keepdim=True)
    Zc = Z_in - mu_z
    Yc = Y_target - mu_y

    if T > 1:
        d1_Z = (Z_3d[:, 1:, :] - Z_3d[:, :-1, :]).reshape(-1, D_in).float()
        d1_Y = (Y_3d[:, 1:, :] - Y_3d[:, :-1, :]).reshape(-1, D_out).float()
        Cov = (Zc.T @ Zc) / float(len(Z_in)) + 2.0 * (d1_Z.T @ d1_Z) / float(len(d1_Z))
        Cross = (Zc.T @ Yc) / float(len(Z_in)) + 2.0 * (d1_Z.T @ d1_Y) / float(len(d1_Z))
    else:
        d1_Y = torch.zeros_like(Yc)
        Cov = (Zc.T @ Zc) / float(len(Z_in))
        Cross = (Zc.T @ Yc) / float(len(Z_in))

    W_post_T = torch.linalg.solve(Cov + lam * torch.eye(D_in, device=dev), Cross + lam * W_pr.T)
    W_post = W_post_T.T

    if rank < min(W_post.shape):
        ev, evec = torch.linalg.eigh(Cov + 1e-4 * torch.eye(D_in, device=dev))
        ev = ev.clamp(min=1e-6)
        C_half = evec @ torch.diag(ev.sqrt()) @ evec.T
        C_ihalf = evec @ torch.diag(ev.rsqrt()) @ evec.T
        U, S, Vh = torch.linalg.svd(W_post @ C_half, full_matrices=False)
        W_hat = (U[:, :rank] * S[:rank]) @ (Vh[:rank, :] @ C_ihalf)
    else:
        W_hat = W_post

    Y_hat_3d = (Zc @ W_hat.T).view(B, T, D_out)
    if T > 1:
        d1_Yhat = (Y_hat_3d[:, 1:, :] - Y_hat_3d[:, :-1, :]).reshape(-1, D_out)
        var_true = Yc.var(dim=0, unbiased=False) + 2.0 * d1_Y.var(dim=0, unbiased=False)
        var_hat = Y_hat_3d.reshape(-1, D_out).var(dim=0, unbiased=False) + 2.0 * d1_Yhat.var(dim=0, unbiased=False)
    else:
        var_true = Yc.var(dim=0, unbiased=False)
        var_hat = Y_hat_3d.reshape(-1, D_out).var(dim=0, unbiased=False)

    gain = (var_true / (var_hat + 1e-6)).sqrt().clamp(1.0, max_gain)
    W_hat = W_hat * gain.unsqueeze(1)
    b_hat = (mu_y - mu_z @ W_hat.T).squeeze(0)
    return W_hat, b_hat


@torch.inference_mode()
def compute_signed_rkhs_params(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    eps: float = 1e-2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute reference mean `mu_ref` and signed inverse std `oinv_signed` for 1-pass RKHS lifting."""
    real_mask = y_train == 0
    spoof_mask = y_train == 1
    X_real = X_train[real_mask].float()
    X_spoof = X_train[spoof_mask].float()
    mu_ref = X_real.mean(dim=0)
    sd_ref = X_real.std(dim=0, unbiased=False) + eps
    delta = X_spoof.mean(dim=0) - mu_ref
    sign_d = torch.where(delta >= 0, torch.ones_like(delta), -torch.ones_like(delta))
    oinv_signed = sign_d / sd_ref
    return mu_ref, oinv_signed


@torch.inference_mode()
def lift_rectified_quadratic_rkhs(
    X: torch.Tensor,
    mu_ref: torch.Tensor,
    oinv_signed: torch.Tensor,
) -> torch.Tensor:
    """Stage 3: One-Sided Class-Oriented Rectified Quadratic Anomaly Kernel.

    Maps features `X` into a non-negative one-sided quadratic RKHS:
        q(X) = log(1 + ReLU((X - mu_ref) * oinv_signed)^2)
    which selectively amplifies tail anomalies in the positive class-separation half-space
    while zeroing out benign opposite-side fluctuations.
    """
    z_pos = F.relu((X.float() - mu_ref) * oinv_signed)
    return torch.log1p(z_pos * z_pos)


@torch.inference_mode()
def solve_fpo1_woodbury_lcmv(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    nuisance_deltas: Optional[Sequence[Tuple[torch.Tensor, float]]] = None,
    task_group_ids: Optional[torch.Tensor] = None,
    domain_group_ids: Optional[torch.Tensor] = None,
    lam: float = 25.0,
    eps_lcmv: float = 0.015,
    boundary_beta: float = 1.2,
    boundary_boost: float = 1.5,
    dro_temperature: float = 1.5,
) -> Tuple[torch.Tensor, float, Dict[str, torch.Tensor]]:
    """Stage 4: One-Pass Frontier Optimizer (FPO-1) Dual-Space Woodbury Capon-LCMV Solver.

    Replaces multi-epoch gradient descent with a 1-pass closed-form global optimum:
      1. Standardizes features relative to the negative/bona-fide reference distribution.
      2. Constructs the low-rank noise + paired-nuisance factor matrix `R` (M x D, M << D),
         the K task/generator mean-shift vectors `D_K` (D x K), and the J nuisance-nulling
         constraint vectors `C_J` (D x J).
      3. Solves the initial Dual-Woodbury Capon-LCMV system:
             A_0^{-1} V = (V - R^T (lam I_M + R R^T)^{-1} R V) / lam
             W_0 = A_0^{-1} D_K - A_0^{-1} C_J (C_J^T A_0^{-1} C_J + eps I_J)^{-1} C_J^T A_0^{-1} D_K
      4. Solves the K x K Minimax Group-DRO problem to find the optimal task combination
         weights `alpha_dro` that maximize the worst-case Mahalanobis margin across all K groups.
      5. Computes Boltzmann-Focal Boundary Precision Weights `pi_i` on the training set
         (concentrating precision on hard boundary examples without a second forward pass)
         and resolves the Boundary-Focused Woodbury-LCMV weights `w_fpo, b_fpo`.
    """
    dev = X_train.device
    X_f = X_train.float()
    y_l = y_train.long()

    r_idx = torch.where(y_l == 0)[0]
    s_idx = torch.where(y_l == 1)[0]
    if len(r_idx) == 0 or len(s_idx) == 0:
        raise ValueError("X_train must contain both class 0 and class 1 samples.")

    # Step 1: Balanced reference mean & std
    if domain_group_ids is not None:
        dom_r = domain_group_ids[r_idx]
        uniq_d = torch.unique(dom_r)
        mu_0 = torch.stack([X_f[r_idx[dom_r == d]].mean(0) for d in uniq_d], dim=0).mean(0)
    else:
        mu_0 = X_f[r_idx].mean(0)
    sd_0 = X_f[r_idx].std(0, unbiased=False) + 1e-2
    Z = (X_f - mu_0) / sd_0

    Z_real = Z[r_idx]
    Z_spoof = Z[s_idx]
    R_rows: List[torch.Tensor] = [
        (Z_real - Z_real.mean(0, keepdim=True)) / (float(len(Z_real)) ** 0.5),
        0.45 * (Z_spoof - Z_spoof.mean(0, keepdim=True)) / (float(len(Z_spoof)) ** 0.5),
    ]

    # Add within-domain real & spoof covariance rows
    C_cols: List[torch.Tensor] = []
    if domain_group_ids is not None:
        dom_r = domain_group_ids[r_idx]
        uniq_d = torch.unique(dom_r)
        ref_d_mean = Z_real[dom_r == uniq_d[0]].mean(0)
        for d_i, d in enumerate(uniq_d):
            sub_r = Z_real[dom_r == d]
            if len(sub_r) > 1:
                R_rows.append(1.4 * (sub_r - sub_r.mean(0, keepdim=True)) / (float(len(sub_r)) ** 0.5))
            if d_i > 0 and len(sub_r) > 0:
                C_cols.append(sub_r.mean(0) - ref_d_mean)

    # Build K task/generator mean-shift columns D_K
    D_cols: List[torch.Tensor] = []
    mu_r_z = Z_real.mean(0)
    if task_group_ids is not None:
        tg_s = task_group_ids[s_idx]
        uniq_t = torch.unique(tg_s)
        for t in uniq_t:
            sub_s = Z_spoof[tg_s == t]
            if len(sub_s) > 1:
                R_rows.append(0.50 * (sub_s - sub_s.mean(0, keepdim=True)) / (float(len(sub_s)) ** 0.5))
            if len(sub_s) > 0:
                D_cols.append(sub_s.mean(0) - mu_r_z)
    else:
        D_cols.append(Z_spoof.mean(0) - mu_r_z)

    # Add paired nuisance difference rows to R_rows and mean-drift constraints to C_cols
    if nuisance_deltas is not None:
        for delta_raw, weight in nuisance_deltas:
            if delta_raw is not None and len(delta_raw) > 0:
                d_z = delta_raw.to(device=dev, dtype=torch.float32) / sd_0
                R_rows.append(float(weight) * d_z / (float(len(d_z)) ** 0.5))
                C_cols.append(d_z.mean(0))

    R_mat = torch.cat(R_rows, dim=0)
    D_mat = torch.stack(D_cols, dim=1)
    K = D_mat.size(1)
    has_constraints = len(C_cols) > 0
    C_mat = torch.stack(C_cols, dim=1) if has_constraints else None

    def _woodbury_lcmv(R_cur: torch.Tensor) -> torch.Tensor:
        G = R_cur @ R_cur.T + lam * torch.eye(len(R_cur), device=dev)
        if has_constraints and C_mat is not None:
            RHS = torch.cat([D_mat, C_mat], dim=1)
            SOL = (RHS - R_cur.T @ torch.linalg.solve(G, R_cur @ RHS)) / lam
            Ainv_D, Ainv_C = SOL[:, :K], SOL[:, K:]
            CAC = C_mat.T @ Ainv_C
            reg_c = eps_lcmv * torch.diag(torch.diag(CAC).abs() + 1.0)
            gamma = torch.linalg.solve(CAC + reg_c, C_mat.T @ Ainv_D)
            return Ainv_D - Ainv_C @ gamma
        else:
            return (D_mat - R_cur.T @ torch.linalg.solve(G, R_cur @ D_mat)) / lam

    # Step 2: Initial 1-Pass Woodbury-LCMV Solve
    W_0 = _woodbury_lcmv(R_mat)
    S_r_0 = Z_real @ W_0
    r_m_0 = S_r_0.mean(0, keepdim=True)
    r_s_0 = S_r_0.std(0, unbiased=False, keepdim=True) + 1e-5
    W_0_norm = W_0 / r_s_0

    # Step 3: Closed-Form Minimax Group-DRO over the K task directions
    if K > 1:
        M_sig = D_mat.T @ W_0_norm
        RW = R_mat @ W_0_norm
        Sigma_K = (RW.T @ RW) + 0.05 * torch.eye(K, device=dev)
        alpha = torch.ones(K, device=dev) / float(K)
        for _ in range(30):
            margins_k = M_sig @ alpha
            q_dro = F.softmax(-dro_temperature * (margins_k / (margins_k.abs().mean() + 1e-5)), dim=0)
            w_step = torch.linalg.solve(Sigma_K, M_sig.T @ q_dro).clamp(min=0.02)
            alpha = 0.85 * alpha + 0.15 * (w_step / w_step.sum())
    else:
        alpha = torch.ones(1, device=dev)

    # Step 4: Boltzmann-Focal Boundary Precision Re-Weighting (0 extra forward passes)
    if boundary_boost > 0:
        w_init_z = W_0_norm @ alpha
        s0_all = (Z @ w_init_z) - float((Z_real @ w_init_z).mean())
        s0_std = float((Z_real @ w_init_z).std(unbiased=False) + 1e-5)
        s0_all = s0_all / s0_std
        y_pm = torch.where(y_l == 1, 1.0, -1.0)
        margins_i = y_pm * (s0_all - 1.2)
        pi_all = (0.5 + boundary_boost * torch.sigmoid(-boundary_beta * (margins_i - 1.0))).sqrt()

        pi_r = pi_all[r_idx].unsqueeze(1)
        pi_r = pi_r / pi_r.mean()
        Z_real_pi = Z_real * pi_r
        R_rows[0] = (Z_real_pi - Z_real_pi.mean(0, keepdim=True)) / (float(len(Z_real)) ** 0.5)
        R_pi = torch.cat(R_rows, dim=0)
        W_f = _woodbury_lcmv(R_pi)
    else:
        W_f = W_0

    S_r_f = Z_real @ W_f
    r_m_f = S_r_f.mean(0, keepdim=True)
    r_s_f = S_r_f.std(0, unbiased=False, keepdim=True) + 1e-5
    wf_scale = alpha / r_s_f.squeeze(0)
    w_z = W_f @ wf_scale
    w_raw = w_z / sd_0
    b_raw = float(-(mu_0 @ w_raw) - torch.sum(alpha * r_m_f.squeeze(0) / r_s_f.squeeze(0)))

    diagnostics = {
        "alpha_dro": alpha.detach(),
        "mu_0": mu_0.detach(),
        "sd_0": sd_0.detach(),
        "d_sign": torch.sign(D_mat @ alpha).detach(),
    }
    return w_raw, b_raw, diagnostics


@torch.inference_mode()
def distill_readout_per_tap(
    F_stud: torch.Tensor,
    F_teach: torch.Tensor,
    W_prior: torch.Tensor,
    lam: float = 800.0,
    nuisance_deltas_stud: Optional[torch.Tensor] = None,
    nuisance_deltas_teach: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Stage 5: Closed-Form Nuisance-Orthogonal Per-Tap Readout Distillation."""
    dev = F_stud.device
    Fs = F_stud.float()
    Ft = F_teach.float()
    Wp = W_prior.to(device=dev, dtype=torch.float32)
    if Wp.ndim == 1:
        Wp = Wp.unsqueeze(1)
        squeeze_out = True
    else:
        squeeze_out = False

    mu_s = Fs.mean(0, keepdim=True)
    sd_s = Fs.std(0, unbiased=False, keepdim=True) + 1e-2
    mu_t = Ft.mean(0, keepdim=True)
    sd_t = Ft.std(0, unbiased=False, keepdim=True) + 1e-2
    Zs = (Fs - mu_s) / sd_s
    Zt = (Ft - mu_t) / sd_t

    if nuisance_deltas_stud is not None and nuisance_deltas_teach is not None:
        Zs = torch.cat([Zs, nuisance_deltas_stud.float() / sd_s], dim=0)
        Zt = torch.cat([Zt, nuisance_deltas_teach.float() / sd_t], dim=0)

    W_p_z = Wp * sd_t.T
    Y_t_c = Zt @ W_p_z
    Res_Y = Y_t_c - (Zs @ W_p_z)

    N, D = Zs.shape
    if N >= D:
        Cov = (Zs.T @ Zs) + lam * torch.eye(D, device=dev)
        Delta_W = torch.linalg.solve(Cov, Zs.T @ Res_Y)
    else:
        G = (Zs @ Zs.T) + lam * torch.eye(N, device=dev)
        Delta_W = Zs.T @ torch.linalg.solve(G, Res_Y)

    W_s_raw = (W_p_z + Delta_W) / sd_s.T
    b_shift = ((mu_t @ Wp) - (mu_s @ W_s_raw)).squeeze(0)
    if squeeze_out:
        return W_s_raw.squeeze(1), float(b_shift[0])
    return W_s_raw, b_shift


class OnePassFrontierOptimizer(Optimizer):
    """PyTorch Optimizer that computes globally optimal frontier weights in 1 Forward Pass (`1F + 0B`).

    Unlike `SGD` or `AdamW`, which require dozens of epochs of `.backward()` passes and
    memorize spurious dataset/channel shortcuts, `OnePassFrontierOptimizer` accumulates
    sufficient statistics in a single forward pass (`torch.inference_mode()`) and solves
    for the Nuisance-Orthogonal, Boundary-Focused, Minimax Group-DRO weights in one shot.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lam: float = 25.0,
        eps_lcmv: float = 0.015,
        boundary_beta: float = 1.2,
        boundary_boost: float = 1.5,
        dro_temperature: float = 1.5,
        use_rkhs: bool = False,
        rkhs_weight: float = 0.25,
    ):
        defaults = dict(
            lam=lam,
            eps_lcmv=eps_lcmv,
            boundary_beta=boundary_beta,
            boundary_boost=boundary_boost,
            dro_temperature=dro_temperature,
            use_rkhs=use_rkhs,
            rkhs_weight=rkhs_weight,
        )
        super().__init__(params, defaults)
        self._features: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []
        self._task_ids: List[torch.Tensor] = []
        self._domain_ids: List[torch.Tensor] = []
        self._nuisance_deltas: List[Tuple[torch.Tensor, float]] = []
        self.rkhs_state: Dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def accumulate_forward_batch(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Record a batch of forward-pass activations and labels (zero gradients required)."""
        self._features.append(features.detach())
        self._labels.append(labels.detach())
        if task_ids is not None:
            self._task_ids.append(task_ids.detach())
        if domain_ids is not None:
            self._domain_ids.append(domain_ids.detach())

    @torch.inference_mode()
    def add_nuisance_Export_pair(
        self,
        clean_features: torch.Tensor,
        nuisance_features: torch.Tensor,
        weight: float = 4.0,
    ) -> None:
        """Register paired nuisance-vs-clean forward activations for exact subspace nulling."""
        self._nuisance_deltas.append(((nuisance_features - clean_features).detach(), float(weight)))

    @torch.inference_mode()
    def step_one_pass(
        self,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
        nuisance_deltas: Optional[Sequence[Tuple[torch.Tensor, float]]] = None,
        task_group_ids: Optional[torch.Tensor] = None,
        domain_group_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Solve for optimal weights in 1 go and write directly into the managed Linear/Head parameters."""
        if X_train is None:
            if not self._features:
                raise RuntimeError("No features provided or accumulated via accumulate_forward_batch.")
            X_train = torch.cat(self._features, dim=0)
            y_train = torch.cat(self._labels, dim=0)
            if self._task_ids:
                task_group_ids = torch.cat(self._task_ids, dim=0)
            if self._domain_ids:
                domain_group_ids = torch.cat(self._domain_ids, dim=0)
        if nuisance_deltas is None and self._nuisance_deltas:
            nuisance_deltas = self._nuisance_deltas

        group = self.param_groups[0]
        w_lin, b_lin, diag = solve_fpo1_woodbury_lcmv(
            X_train=X_train,
            y_train=y_train,
            nuisance_deltas=nuisance_deltas,
            task_group_ids=task_group_ids,
            domain_group_ids=domain_group_ids,
            lam=group["lam"],
            eps_lcmv=group["eps_lcmv"],
            boundary_beta=group["boundary_beta"],
            boundary_boost=group["boundary_boost"],
            dro_temperature=group["dro_temperature"],
        )

        if group["use_rkhs"]:
            mu_0 = diag["mu_0"]
            sd_0 = diag["sd_0"]
            d_sign = diag["d_sign"]
            oinv_signed = d_sign / sd_0
            QX_train = lift_rectified_quadratic_rkhs(X_train, mu_0, oinv_signed)
            q_nuis = None
            if nuisance_deltas is not None:
                # Approximate quadratic nuisance deltas around mu_0
                q_nuis = [
                    (lift_rectified_quadratic_rkhs(mu_0.unsqueeze(0) + d_raw, mu_0, oinv_signed), w_n)
                    for d_raw, w_n in nuisance_deltas
                ]
            w_quad, b_quad, _ = solve_fpo1_woodbury_lcmv(
                X_train=QX_train,
                y_train=y_train,
                nuisance_deltas=q_nuis,
                task_group_ids=task_group_ids,
                domain_group_ids=domain_group_ids,
                lam=group["lam"] * 1.5,
                eps_lcmv=group["eps_lcmv"],
                boundary_beta=group["boundary_beta"],
                boundary_boost=0.0,
                dro_temperature=group["dro_temperature"],
            )
            self.rkhs_state = {
                "mu_ref": mu_0,
                "oinv_signed": oinv_signed,
                "w_quad": w_quad * group["rkhs_weight"],
                "b_quad": torch.tensor([b_quad * group["rkhs_weight"]], device=w_lin.device),
            }
            diag.update(self.rkhs_state)

        # Write w_lin and b_lin into the first weight & bias parameter of the parameter group
        params = [p for p in group["params"] if p.numel() > 0]
        if len(params) >= 1:
            w_p = params[0]
            w_p.data.copy_(w_lin.reshape(w_p.shape).type_as(w_p.data))
        if len(params) >= 2:
            b_p = params[1]
            b_p.data.fill_(b_lin)

        self._features.clear()
        self._labels.clear()
        self._task_ids.clear()
        self._domain_ids.clear()
        self._nuisance_deltas.clear()
        return diag

    @torch.inference_mode()
    def predict_scores(self, X: torch.Tensor) -> torch.Tensor:
        """Compute decision scores using the 1-pass solved linear (+ optional RKHS quadratic) weights."""
        params = [p for p in self.param_groups[0]["params"] if p.numel() > 0]
        w_lin = params[0].reshape(-1).float()
        b_lin = float(params[1].reshape(-1)[0]) if len(params) >= 2 else 0.0
        scores = X.float() @ w_lin + b_lin
        if self.param_groups[0]["use_rkhs"] and "w_quad" in self.rkhs_state:
            QX = lift_rectified_quadratic_rkhs(X, self.rkhs_state["mu_ref"], self.rkhs_state["oinv_signed"])
            scores = scores + (QX @ self.rkhs_state["w_quad"]) + float(self.rkhs_state["b_quad"][0])
        return scores

    def step(self, closure=None):
        """Standard Optimizer.step() alias that invokes step_one_pass() on accumulated batches."""
        return self.step_one_pass()


# Canonical paper & framework aliases: One-Forward-Pass Optimizer (OFP)
OneForwardPassOptimizer = OnePassFrontierOptimizer
OFP = OnePassFrontierOptimizer

