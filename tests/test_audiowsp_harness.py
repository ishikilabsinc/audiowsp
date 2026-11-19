"""Unit tests for the AudioWSP + OFP + OT-SVD harness, Anti-Cheating Eval Firewall, and ASR scoring."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from harness.asr import (
    asr_counts_as_improvement,
    asr_report,
    char_error_rate,
    ctc_greedy_decode,
    normalize_asr_text,
    word_error_rate,
)
from harness.audiowsp import (
    AcousticNuisanceProfiler,
    OTVelocityCompressor,
    OneForwardPassOptimizer,
    SubspaceNullingSurgery,
    solve_ofp_ctc_readout,
    solve_ranpac_ctc_baseline,
    solve_ridge_ctc_baseline,
)
from harness.firewall import (
    EvalFirewallViolation,
    assert_practice_path,
    assert_practice_split,
    compute_calibration_audit_certificate,
)


def _assert_raises_firewall(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except EvalFirewallViolation:
        return
    raise AssertionError(f"Expected EvalFirewallViolation from {fn.__name__}({args}, {kwargs})")


def test_eval_firewall_blocks_test_splits_and_paths():
    # Allowed practice/dev splits must pass
    for allowed in ("train", "dev", "dev-clean", "dev-other", "validation.clean", "practice"):
        assert assert_practice_split(allowed) == allowed.lower()

    # Locked evaluation splits must immediately raise EvalFirewallViolation
    for forbidden in (
        "test",
        "test-clean",
        "test-other",
        "test.clean",
        "test.other",
        "tedlium-test",
        "mls-test",
        "eval",
        "asvspoof2021_la_eval",
        "inthewild",
        "decro",
    ):
        _assert_raises_firewall(assert_practice_split, forbidden)

    # Locked evaluation file paths must immediately raise EvalFirewallViolation
    for bad_path in (
        "/data/librispeech/test-clean/0000.parquet",
        "/data/librispeech/other/test/0000.parquet",
        "/data/tedlium/test-00000-of-00001.parquet",
        "/data/asvspoof5.eval.track_1.tsv",
    ):
        _assert_raises_firewall(assert_practice_path, bad_path)

    # Audit certificate works for dev splits and fails on test splits
    cert = compute_calibration_audit_certificate(
        task="asr_multidataset",
        practice_splits=["dev-clean", "dev-other"],
        utterance_ids=["utt_001", "utt_002"],
        nuisance_families=["telephony_8khz", "room_reverb"],
    )
    assert cert["firewall_verified"] is True
    assert cert["zero_eval_leakage"] is True
    assert cert["n_calibration_utterances"] == 2


def test_acoustic_nuisance_profiler_and_task_orthogonal_subspace():
    profiler = AcousticNuisanceProfiler(sample_rate=16000, seed=7)
    t = np.linspace(0.0, 1.0, 16000, endpoint=False, dtype=np.float32)
    wav = 0.5 * np.sin(2.0 * np.pi * 440.0 * t) + 0.2 * np.sin(2.0 * np.pi * 1200.0 * t)

    # Firewall blocks generating probes from test-clean
    _assert_raises_firewall(profiler.generate_paired_probes, wav, split="test-clean")

    probes = profiler.generate_paired_probes(wav, split="dev-clean")
    assert set(probes.keys()) == set(AcousticNuisanceProfiler.NUISANCE_FAMILIES)
    for name, p_wav in probes.items():
        assert p_wav.shape == wav.shape
        assert np.isfinite(p_wav).all()

    # Check task-orthogonal subspace extraction
    torch.manual_seed(11)
    D = 64
    task_basis = torch.linalg.qr(torch.randn(D, 4), mode="reduced")[0].T  # (4, D) orthonormal
    clean_acts = torch.randn(32, 20, D)
    nuis_dir = torch.randn(2, D)
    pert_acts = clean_acts + torch.randn(32, 20, 2) @ nuis_dir

    U_nuis, stats = AcousticNuisanceProfiler.extract_task_orthogonal_nuisance_subspace(
        clean_acts, pert_acts, task_basis=task_basis, rank_nuis=6
    )
    assert U_nuis.shape == (6, D)
    # Verify U_nuis is strictly orthogonal to task_basis
    overlap = torch.linalg.norm(U_nuis @ task_basis.T).item()
    assert overlap < 1e-5, f"Expected task-orthogonal overlap < 1e-5, got {overlap}"
    assert stats["drift_ratio"] > 0.0


def test_ofp_ctc_readout_beats_ridge_and_ranpac_under_nuisance_shift():
    torch.manual_seed(23)
    N, D, V = 1200, 96, 32
    W_true = torch.randn(V, D) * 0.25
    b_true = torch.randn(V) * 0.1

    # Low-rank acoustic nuisance subspace (4 dims)
    U_nuis = torch.linalg.qr(torch.randn(D, 4), mode="reduced")[0]  # (D, 4)
    # Make W_true orthogonal to U_nuis so clean phonetic target depends on D \ U_nuis
    W_true = W_true @ (torch.eye(D) - U_nuis @ U_nuis.T)
    # Corrupted prior weight has nuisance leakage
    W_prior = W_true + 0.5 * (torch.randn(V, 4) @ U_nuis.T)
    b_prior = b_true.clone()

    H_clean = torch.randn(N, D)
    # Confident teacher targets on practice frames + spurious correlation with nuisance on practice set
    spurious_z = torch.randn(N, 4)
    H_prac = H_clean + spurious_z @ U_nuis.T
    Y_target = H_clean @ W_true.T + b_true + 0.35 * (spurious_z @ torch.randn(4, V))

    # Paired acoustic nuisance probes on practice set
    H_pert_1 = H_prac + (torch.randn(N, 4) * 2.5) @ U_nuis.T
    H_pert_2 = H_prac + (torch.randn(N, 4) * 2.5) @ U_nuis.T

    # Firewall blocks test-other
    _assert_raises_firewall(
        solve_ofp_ctc_readout, H_prac, Y_target, W_prior, b_prior, split="test-other"
    )

    W_ofp, b_ofp, _ = solve_ofp_ctc_readout(
        H_prac,
        Y_target,
        W_prior,
        b_prior,
        H_nuis_list=[(H_pert_1, 1.0), (H_pert_2, 1.0)],
        lam_prior=2.0,
        gamma_nuis=15.0,
        boundary_beta=1.2,
        boundary_boost=1.5,
        split="dev-clean",
    )
    W_ridge, b_ridge = solve_ridge_ctc_baseline(
        H_prac, Y_target, W_prior, b_prior, lam=2.0, split="dev-clean"
    )
    R_rp, W_rp, b_rp = solve_ranpac_ctc_baseline(
        H_prac, Y_target, proj_dim=192, lam=5.0, split="dev-clean"
    )

    # Evaluate on unseen test frames with severe acoustic nuisance drift along U_nuis
    H_eval_clean = torch.randn(800, D)
    Y_eval_true = H_eval_clean @ W_true.T + b_true
    H_eval_noisy = H_eval_clean + (torch.randn(800, 4) * 4.0) @ U_nuis.T

    mse_prior = ((H_eval_noisy @ W_prior.T + b_prior - Y_eval_true) ** 2).mean().item()
    mse_ridge = ((H_eval_noisy @ W_ridge.T + b_ridge - Y_eval_true) ** 2).mean().item()
    mse_ranpac = ((torch.relu(H_eval_noisy @ R_rp) @ W_rp.T + b_rp - Y_eval_true) ** 2).mean().item()
    mse_ofp = ((H_eval_noisy @ W_ofp.T + b_ofp - Y_eval_true) ** 2).mean().item()

    assert mse_ofp < mse_prior, f"Expected OFP MSE ({mse_ofp:.4f}) < Prior ({mse_prior:.4f})"
    assert mse_ofp < mse_ridge, f"Expected OFP MSE ({mse_ofp:.4f}) < Ridge ({mse_ridge:.4f})"
    assert mse_ofp < mse_ranpac, f"Expected OFP MSE ({mse_ofp:.4f}) < RanPAC ({mse_ranpac:.4f})"


def test_ot_svd_preserves_velocity_better_than_standard_svd_and_asvd():
    torch.manual_seed(31)
    B, T, D_in, D_out, rank = 16, 40, 64, 64, 16
    # Time-varying sequence with rapid frame-to-frame consonant spikes
    base_seq = torch.randn(B, T, D_in) * 0.4
    spikes = torch.randn(B, T, D_in) * (torch.rand(B, T, 1) > 0.8).float() * 2.5
    Z_3d = base_seq + spikes

    W_prior = torch.randn(D_out, D_in) / np.sqrt(D_in)
    b_prior = torch.randn(D_out) * 0.1
    Y_3d = Z_3d @ W_prior.T + b_prior

    U_ot, V_ot, b_ot = OTVelocityCompressor.compress_ot_svd(
        Z_3d, Y_3d, W_prior, b_prior, rank=rank, split="dev-clean"
    )
    U_std, V_std, b_std = OTVelocityCompressor.compress_standard_svd(W_prior, b_prior, rank=rank)
    U_asvd, V_asvd, b_asvd = OTVelocityCompressor.compress_asvd(
        Z_3d, W_prior, b_prior, rank=rank, split="dev-clean"
    )

    Y_ot = (Z_3d @ V_ot.T) @ U_ot.T + b_ot
    Y_std = (Z_3d @ V_std.T) @ U_std.T + b_std
    Y_asvd = (Z_3d @ V_asvd.T) @ U_asvd.T + b_asvd

    # Compare frame-to-frame transition velocity variance preservation: Var(d1 Y_hat) / Var(d1 Y)
    d1_true_var = (Y_3d[:, 1:] - Y_3d[:, :-1]).var(dim=(0, 1))
    d1_ot_var = (Y_ot[:, 1:] - Y_ot[:, :-1]).var(dim=(0, 1))
    d1_std_var = (Y_std[:, 1:] - Y_std[:, :-1]).var(dim=(0, 1))
    d1_asvd_var = (Y_asvd[:, 1:] - Y_asvd[:, :-1]).var(dim=(0, 1))

    err_vel_ot = (d1_ot_var - d1_true_var).abs().mean().item()
    err_vel_std = (d1_std_var - d1_true_var).abs().mean().item()
    err_vel_asvd = (d1_asvd_var - d1_true_var).abs().mean().item()

    assert err_vel_ot < err_vel_std * 0.6
    assert err_vel_ot < err_vel_asvd * 0.6


def test_asr_scoring_and_ctc_decoding():
    assert normalize_asr_text("Hello, World! Don 't <unk>") == "HELLO WORLD DON'T"
    assert math.isclose(word_error_rate(["THE CAT SAT ON THE MAT"], ["THE CAT ON THE MAT"]), 1.0 / 6.0)
    assert math.isclose(char_error_rate(["CAT"], ["BAT"]), 1.0 / 3.0)

    # Synthetic CTC sequence for "HI" -> H (11), H (11), blank (0), I (10)
    logits = np.full((4, 32), -10.0, dtype=np.float32)
    logits[0, 11] = 5.0
    logits[1, 11] = 5.0
    logits[2, 0] = 5.0
    logits[3, 10] = 5.0
    assert ctc_greedy_decode(logits) == "HI"

    base_rows = [
        {"ref": "THE CAT SAT ON THE MAT", "hyp": "THE CAT ON MAT", "split": "test-other"},
        {"ref": "HELLO WORLD", "hyp": "HELLO WORLD", "split": "test-clean"},
    ]
    edit_rows = [
        {"ref": "THE CAT SAT ON THE MAT", "hyp": "THE CAT SAT ON THE MAT", "split": "test-other"},
        {"ref": "HELLO WORLD", "hyp": "HELLO WORLD", "split": "test-clean"},
    ]
    rep_b = asr_report(base_rows)
    rep_e = asr_report(edit_rows)
    dec = asr_counts_as_improvement(rep_b, rep_e)
    assert dec["counts"] is True


if __name__ == "__main__":
    test_eval_firewall_blocks_test_splits_and_paths()
    test_acoustic_nuisance_profiler_and_task_orthogonal_subspace()
    test_ofp_ctc_readout_beats_ridge_and_ranpac_under_nuisance_shift()
    test_ot_svd_preserves_velocity_better_than_standard_svd_and_asvd()
    test_asr_scoring_and_ctc_decoding()
    print("ALL AUDIOWSP + OFP + OT-SVD + FIREWALL UNIT TESTS PASSED!")
