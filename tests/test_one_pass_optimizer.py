import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.one_pass_frontier_optimizer import (
    OnePassFrontierOptimizer,
    distill_readout_per_tap,
    lift_rectified_quadratic_rkhs,
    solve_fpo1_woodbury_lcmv,
    solve_ot_variance_preserving_svd,
)


def test_fpo1_beats_adamw_under_spurious_shortcut_and_nuisance_shift():
    """Verify FPO-1 (1 forward pass, 0 backward passes) beats 25-epoch AdamW under shortcut shift."""
    torch.manual_seed(42)
    N_tr, N_te, D = 400, 400, 256

    # True causal signal lives in dims 0..15; high-variance spurious channel shortcut lives in dims 16..47
    y_tr = torch.cat([torch.zeros(N_tr // 2), torch.ones(N_tr // 2)]).long()
    y_te = torch.cat([torch.zeros(N_te // 2), torch.ones(N_te // 2)]).long()

    y_tr_pm = torch.where(y_tr == 1, 1.0, -1.0).unsqueeze(1)
    y_te_pm = torch.where(y_te == 1, 1.0, -1.0).unsqueeze(1)

    X_tr = 0.5 * torch.randn(N_tr, D)
    X_tr[:, :16] += 0.65 * y_tr_pm
    # Spurious shortcut in training: strongly correlated with y_tr with 4x larger amplitude!
    X_tr[:, 16:48] += 2.5 * y_tr_pm + 1.5 * torch.randn(N_tr, 32)

    # Paired nuisance clips (e.g. codec/channel shift on dims 16..47)
    clean_p = 0.5 * torch.randn(80, D)
    nuis_p = clean_p.clone()
    nuis_p[:, 16:48] += 3.0 * torch.randn(80, 32) + 2.0

    # Unseen test domain: spurious shortcut in dims 16..47 is inverted/scrambled, only causal dims 0..15 hold!
    X_te = 0.5 * torch.randn(N_te, D)
    X_te[:, :16] += 0.65 * y_te_pm
    X_te[:, 16:48] -= 2.0 * y_te_pm + 1.5 * torch.randn(N_te, 32)

    # 1. Train classical AdamW for 25 epochs with backpropagation
    head_adam = nn.Linear(D, 1)
    nn.init.zeros_(head_adam.weight)
    nn.init.zeros_(head_adam.bias)
    opt_adam = torch.optim.AdamW(head_adam.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(25):
        logits = head_adam(X_tr).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr.float())
        opt_adam.zero_grad()
        loss.backward()
        opt_adam.step()

    with torch.inference_mode():
        preds_adam = (head_adam(X_te).squeeze(1) > 0.0).long()
        acc_adam = (preds_adam == y_te).float().mean().item()

    # 2. Solve in 1 Forward Pass (0 backward passes) with OnePassFrontierOptimizer (FPO-1)
    head_fpo = nn.Linear(D, 1)
    nn.init.zeros_(head_fpo.weight)
    nn.init.zeros_(head_fpo.bias)
    opt_fpo = OnePassFrontierOptimizer(head_fpo.parameters(), lam=10.0, use_rkhs=True)
    opt_fpo.accumulate_forward_batch(X_tr, y_tr)
    opt_fpo.add_nuisance_Export_pair(clean_p, nuis_p, weight=8.0)
    opt_fpo.step()

    with torch.inference_mode():
        scores_fpo = opt_fpo.predict_scores(X_te)
        preds_fpo = (scores_fpo > scores_fpo.median()).long()
        acc_fpo = (preds_fpo == y_te).float().mean().item()

    assert acc_fpo > 0.90, f"Expected FPO-1 unseen accuracy > 90%, got {acc_fpo:.3f}"
    assert acc_fpo > acc_adam + 0.30, f"Expected FPO-1 ({acc_fpo:.3f}) to crush AdamW ({acc_adam:.3f})"


def test_ot_variance_preserving_svd_preserves_temporal_d1_energy():
    """Verify solve_ot_variance_preserving_svd preserves output and frame-to-frame d1 variance."""
    torch.manual_seed(7)
    B, T, D_in, D_out = 16, 40, 64, 64
    Z = torch.randn(B, T, D_in)
    W_true = torch.randn(D_out, D_in) * 0.25
    b_true = torch.randn(D_out) * 0.1
    Y = F.linear(Z, W_true, b_true) + 0.05 * torch.randn(B, T, D_out)

    W_hat, b_hat = solve_ot_variance_preserving_svd(
        Z, Y, W_prior=torch.zeros_like(W_true), b_prior=torch.zeros_like(b_true), rank=16, lam=1e-2
    )
    Y_hat = F.linear(Z, W_hat, b_hat)
    d1_Y = (Y[:, 1:] - Y[:, :-1]).reshape(-1, D_out)
    d1_Yhat = (Y_hat[:, 1:] - Y_hat[:, :-1]).reshape(-1, D_out)

    e_true = Y.reshape(-1, D_out).var(0) + 2.0 * d1_Y.var(0)
    e_hat = Y_hat.reshape(-1, D_out).var(0) + 2.0 * d1_Yhat.var(0)
    ratio = (e_hat / e_true).mean().item()
    assert 0.85 <= ratio <= 1.15, f"Expected OT variance ratio near 1.0, got {ratio:.3f}"


def test_distill_readout_per_tap():
    """Verify distill_readout_per_tap recovers teacher scores from perturbed student features."""
    torch.manual_seed(11)
    N, D = 200, 128
    F_t = torch.randn(N, D)
    R_rot = torch.linalg.qr(torch.randn(D, D))[0]
    F_s = F_t @ R_rot + 0.05 * torch.randn(N, D)
    w_prior = torch.randn(D)

    w_s, b_s = distill_readout_per_tap(F_s, F_t, w_prior, lam=5.0)
    y_t = F_t @ w_prior
    y_s = F_s @ w_s + b_s
    corr = torch.corrcoef(torch.stack([y_t, y_s]))[0, 1].item()
    assert corr > 0.95, f"Expected distilled score correlation > 0.95, got {corr:.4f}"
