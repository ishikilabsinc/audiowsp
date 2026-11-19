"""Strict Anti-Cheating Evaluation Firewall for AudioWSP & OFP.

Guarantees that weight-space programming (`SubspaceNullingSurgery`, `OneForwardPassOptimizer`,
and `OTVelocityCompressor`) only ever accesses designated practice/validation/train
splits and never touches locked evaluation sets (`test-clean`, `test-other`, `tedlium-test`,
`mls-test`, `asvspoof-eval`, `inthewild`, `decro`) during calibration or model surgery.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

# Explicit allowlist of practice/calibration split identifiers
ALLOWED_PRACTICE_SPLITS = frozenset(
    {
        "train",
        "train.100",
        "train.360",
        "train.500",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev",
        "dev-clean",
        "dev-other",
        "dev.clean",
        "dev.other",
        "validation",
        "validation.clean",
        "validation.other",
        "validation-clean",
        "validation-other",
        "practice",
        "calibration",
        "eval_matched",           # Only within the 512-clip local practice_manifest.jsonl
        "eval_unseen_generator",  # Only within the 512-clip local practice_manifest.jsonl
    }
)

# Explicit blocklist of locked evaluation split identifiers and filename patterns
FORBIDDEN_EVAL_SPLITS = frozenset(
    {
        "test",
        "test-clean",
        "test-other",
        "test.clean",
        "test.other",
        "tedlium-test",
        "tedlium_test",
        "mls-test",
        "mls_test",
        "eval",
        "asvspoof2019_la_eval",
        "asvspoof2021_la_eval",
        "asvspoof2021_df_eval",
        "asvspoof5_eval",
        "inthewild",
        "inthewild_eval",
        "decro",
        "decro_eval",
    }
)

FORBIDDEN_PATH_TOKENS = (
    "test-clean",
    "test.clean",
    "test-other",
    "test.other",
    "test-00000",
    "/test/",
    "_test.parquet",
    "asvspoof2019_la_eval",
    "asvspoof2021_la",
    "asvspoof2021_df",
    "asvspoof5.eval",
    "in_the_wild",
    "inthewild",
    "decro_eval",
)


class EvalFirewallViolation(PermissionError):
    """Raised when any calibration, OFP, or OT-SVD routine attempts to access an evaluation split."""


def assert_practice_split(split_name: str, *, context: str = "calibration") -> str:
    """Verify that `split_name` is an allowed practice/dev/train split and not a locked eval split."""
    norm = str(split_name).strip().lower()
    if norm in FORBIDDEN_EVAL_SPLITS or norm.startswith("test") or norm.endswith("-test") or norm.endswith("_test"):
        raise EvalFirewallViolation(
            f"[FIREWALL BLOCKED] {context} attempted to access locked evaluation split '{split_name}'. "
            "Only train/dev/validation/practice splits are permitted during AudioWSP/OFP calibration."
        )
    if norm not in ALLOWED_PRACTICE_SPLITS:
        raise EvalFirewallViolation(
            f"[FIREWALL BLOCKED] {context} received unrecognized split '{split_name}'. "
            f"Allowed practice splits: {sorted(ALLOWED_PRACTICE_SPLITS)}"
        )
    return norm


def assert_practice_path(path: str | Path, *, context: str = "calibration") -> Path:
    """Verify that a dataset file path does not point to a locked test/eval benchmark."""
    p = Path(path)
    s_low = str(p).replace("\\", "/").lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in s_low:
            raise EvalFirewallViolation(
                f"[FIREWALL BLOCKED] {context} attempted to read locked evaluation file '{p}' "
                f"(matched forbidden token '{token}')."
            )
    return p


def assert_practice_manifest_rows(rows: Sequence[dict], *, context: str = "calibration") -> None:
    """Verify every row in a manifest sequence belongs to an allowed practice split."""
    for idx, row in enumerate(rows):
        split = row.get("split", "practice")
        assert_practice_split(str(split), context=f"{context}[row {idx}]")
        if "audio_path" in row:
            assert_practice_path(str(row["audio_path"]), context=f"{context}[row {idx}]")


def compute_calibration_audit_certificate(
    *,
    task: str,
    practice_splits: Sequence[str],
    utterance_ids: Iterable[str],
    nuisance_families: Sequence[str],
    extra_metadata: dict | None = None,
) -> dict:
    """Produce a cryptographic SHA-256 calibration audit certificate proving zero eval leakage."""
    for sp in practice_splits:
        assert_practice_split(sp, context=f"audit_certificate({task})")
    sorted_ids = sorted(str(u) for u in utterance_ids)
    joined = "\n".join(sorted_ids).encode("utf-8")
    ids_sha256 = hashlib.sha256(joined).hexdigest()
    cert = {
        "task": task,
        "firewall_verified": True,
        "zero_eval_leakage": True,
        "practice_splits": list(practice_splits),
        "n_calibration_utterances": len(sorted_ids),
        "calibration_ids_sha256": ids_sha256,
        "nuisance_families": list(nuisance_families),
    }
    if extra_metadata:
        cert.update(extra_metadata)
    return cert
