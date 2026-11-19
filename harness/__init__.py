"""Unified Open-Source Harness for Weight-Space Programming (AudioWSP) & One-Forward-Pass Optimization (OFP) on Audio Models.

Supports both Audio Deepfake / Spoof Detection (`exp1_unseen_generators`) and
Automatic Speech Recognition (`exp2_asr_generalization`) under a single zero-leakage
evaluation firewall (`1 Forward Pass + 0 Backward Passes`).
"""

from __future__ import annotations

__version__ = "0.5.0"

from harness.firewall import (
    ALLOWED_PRACTICE_SPLITS,
    FORBIDDEN_EVAL_SPLITS,
    EvalFirewallViolation,
    assert_practice_manifest_rows,
    assert_practice_path,
    assert_practice_split,
    compute_calibration_audit_certificate,
)

__all__ = [
    "__version__",
    "ALLOWED_PRACTICE_SPLITS",
    "FORBIDDEN_EVAL_SPLITS",
    "EvalFirewallViolation",
    "assert_practice_manifest_rows",
    "assert_practice_path",
    "assert_practice_split",
    "compute_calibration_audit_certificate",
]
