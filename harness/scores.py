"""Shortcut report for an audio deepfake detector score.

The score is a scalar. Larger means the clip is more likely to be synthetic
speech. Label 1 is synthetic and label 0 is real. Equal error rate is the
headline because a monotone remapping of the score does not change it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SPLIT_MATCHED = "eval_matched"
SPLIT_UNSEEN = "eval_unseen_generator"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def equal_error_rate(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    bona = np.sort(scores[labels == 0])
    spoof = np.sort(scores[labels == 1])
    n_b, n_s = len(bona), len(spoof)
    if n_b == 0 or n_s == 0:
        raise ValueError("both classes are required")
    thresholds = np.unique(scores)
    candidates = np.concatenate([thresholds, [float(thresholds[-1]) + 1.0]])
    false_accept = (n_b - np.searchsorted(bona, candidates, side="left")) / n_b
    false_reject = np.searchsorted(spoof, candidates, side="left") / n_s
    gaps = np.abs(false_accept - false_reject)
    rates = 0.5 * (false_accept + false_reject)
    min_gap = float(np.min(gaps))
    mask = gaps <= min_gap + 1e-12
    return float(np.min(rates[mask]))


def roc_auc(scores, labels) -> float:
    """AUC for label 1 as the positive class. Ties count as 0.5."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes are required")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    n = len(scores)
    change = np.concatenate([[True], sorted_scores[1:] != sorted_scores[:-1], [True]])
    starts = np.flatnonzero(change)
    group_idx = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    avg_ranks = 0.5 * (starts[:-1] + 1 + starts[1:])
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg_ranks[group_idx]
    sum_pos_ranks = float(np.sum(ranks[labels == 1]))
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _column(rows: list[dict], key: str):
    return [row.get(key) for row in rows]


def _finite_pair(scores, labels) -> bool:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return bool(np.isfinite(scores).all() and (labels == 0).any() and (labels == 1).any())


def _binary_probe_auc(scores, attribute_positive: np.ndarray) -> float | None:
    """AUC of the score ranking a binary attribute. None when one side is empty."""
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(attribute_positive, dtype=bool)
    if positive.all() or not positive.any():
        return None
    # Reuse roc_auc by treating the attribute as the label.
    labels = positive.astype(np.int64)
    return roc_auc(scores, labels)


def speaker_dependence(scores, speakers) -> float | None:
    """Share of score variance explained by speaker. 0 means none, 1 means all."""
    scores = np.asarray(scores, dtype=np.float64)
    speakers = np.asarray(speakers)
    if len(scores) < 2 or len(np.unique(speakers)) < 2:
        return None
    centered = scores - float(np.mean(scores))
    total = float(np.dot(centered, centered))
    if total == 0.0:
        return 0.0
    between = 0.0
    for speaker in np.unique(speakers):
        group = scores[speakers == speaker]
        between += len(group) * (float(np.mean(group)) - float(np.mean(scores))) ** 2
    return float(between / total)


def same_transcript_eer(rows: list[dict]) -> float | None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        transcript = row.get("transcript_id")
        if transcript is None:
            continue
        grouped.setdefault(str(transcript), []).append(row)
    pooled = []
    for group in grouped.values():
        labels = {int(item["label"]) for item in group}
        if labels == {0, 1}:
            pooled.extend(group)
    if not pooled:
        return None
    scores = [float(item["score"]) for item in pooled]
    labels = [int(item["label"]) for item in pooled]
    if not _finite_pair(scores, labels):
        return None
    return equal_error_rate(scores, labels)


def cross_transcript_correlation(rows: list[dict]) -> float | None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        transcript = row.get("transcript_id")
        if transcript is None:
            continue
        grouped.setdefault(str(transcript), []).append(row)
    bona_means = []
    spoof_means = []
    for group in grouped.values():
        bona = [float(item["score"]) for item in group if int(item["label"]) == 0]
        spoof = [float(item["score"]) for item in group if int(item["label"]) == 1]
        if bona and spoof:
            bona_means.append(float(np.mean(bona)))
            spoof_means.append(float(np.mean(spoof)))
    if len(bona_means) < 3:
        return None
    bona_array = np.asarray(bona_means, dtype=np.float64)
    spoof_array = np.asarray(spoof_means, dtype=np.float64)
    if float(np.std(bona_array)) == 0.0 or float(np.std(spoof_array)) == 0.0:
        return None
    return float(np.corrcoef(bona_array, spoof_array)[0, 1])


def _class_block(rows: list[dict]) -> dict:
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    block = {"n": int(len(rows)), "n_real": int(np.sum(labels == 0)), "n_spoof": int(np.sum(labels == 1))}
    if _finite_pair(scores, labels):
        block["eer"] = equal_error_rate(scores, labels)
        block["auc"] = roc_auc(scores, labels)
    else:
        block["eer"] = None
        block["auc"] = None
    return block


def _probe(rows: list[dict], key: str, positive_value: str) -> float | None:
    bona = [row for row in rows if int(row["label"]) == 0 and row.get(key) is not None]
    if len(bona) < 2:
        return None
    scores = [float(row["score"]) for row in bona]
    flags = [str(row.get(key)) == positive_value for row in bona]
    return _binary_probe_auc(scores, np.asarray(flags))


def language_probe(rows: list[dict]) -> dict:
    bona = [row for row in rows if int(row["label"]) == 0 and row.get("language")]
    languages = sorted({str(row["language"]) for row in bona})
    if len(languages) != 2:
        return {"positive_language": None, "auc": None}
    positive = languages[0]
    return {"positive_language": positive, "auc": _probe(bona, "language", positive)}


def report(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no score rows")
    for row in rows:
        if "score" not in row or "label" not in row:
            raise ValueError("each row needs score and label")
        if int(row["label"]) not in (0, 1):
            raise ValueError("label must be 0 (real) or 1 (synthetic)")
    overall = _class_block(rows)
    by_split = {}
    for split in sorted({str(row["split"]) for row in rows if row.get("split")}):
        by_split[split] = _class_block([row for row in rows if str(row.get("split")) == split])
    by_generator = {}
    for generator in sorted({str(row["generator"]) for row in rows if row.get("generator")}):
        by_generator[generator] = _class_block(
            [row for row in rows if str(row.get("generator")) == generator]
        )
    bona = [row for row in rows if int(row["label"]) == 0 and row.get("speaker") is not None]
    dependence = None
    if bona:
        dependence = speaker_dependence(
            [float(row["score"]) for row in bona],
            [str(row["speaker"]) for row in bona],
        )
    language = language_probe(rows)
    return {
        "n": overall["n"],
        "eer": overall["eer"],
        "auc": overall["auc"],
        "by_split": by_split,
        "by_generator": by_generator,
        "gender_probe_auc_bona_fide": _probe(rows, "gender", "f"),
        "language_probe_auc_bona_fide": language["auc"],
        "language_probe_positive": language["positive_language"],
        "speaker_dependence_bona_fide": dependence,
        "same_transcript_eer": same_transcript_eer(rows),
        "cross_transcript_score_correlation": cross_transcript_correlation(rows),
    }


def _split_eer(summary: dict, split: str) -> float | None:
    block = summary.get("by_split", {}).get(split)
    if not block:
        return None
    return block.get("eer")


def _distance_from_half(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return abs(float(value) - 0.5)


def _higher(new, old, tol: float) -> bool:
    if new is None or old is None:
        return False
    if isinstance(new, float) and math.isnan(new):
        return False
    if isinstance(old, float) and math.isnan(old):
        return False
    return float(new) > float(old) + tol


def counts_as_improvement(base: dict, edited: dict, tol: float = 0.02) -> dict:
    """An edit counts when unseen-generator EER falls and shortcuts do not get worse."""
    reasons = []
    base_unseen = _split_eer(base, SPLIT_UNSEEN)
    edited_unseen = _split_eer(edited, SPLIT_UNSEEN)
    if base_unseen is None or edited_unseen is None:
        reasons.append("unseen-generator split is missing")
    elif not edited_unseen < base_unseen - 1e-9:
        reasons.append("unseen-generator EER did not fall")
    base_matched = _split_eer(base, SPLIT_MATCHED)
    edited_matched = _split_eer(edited, SPLIT_MATCHED)
    if base_matched is None or edited_matched is None:
        reasons.append("matched split is missing")
    elif _higher(edited_matched, base_matched, tol):
        reasons.append("matched EER got worse")
    comparisons = (
        ("gender probe moved away from 0.5", "gender_probe_auc_bona_fide", True),
        ("language probe moved away from 0.5", "language_probe_auc_bona_fide", True),
        ("speaker dependence rose", "speaker_dependence_bona_fide", False),
        ("same-transcript EER got worse", "same_transcript_eer", False),
        ("cross-transcript score correlation rose", "cross_transcript_score_correlation", False),
    )
    for message, key, center_half in comparisons:
        if center_half:
            new = _distance_from_half(edited.get(key))
            old = _distance_from_half(base.get(key))
        else:
            new = edited.get(key)
            old = base.get(key)
        if _higher(new, old, tol):
            reasons.append(message)
    return {"counts": not reasons, "reasons": reasons}
