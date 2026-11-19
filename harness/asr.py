"""Automatic Speech Recognition (ASR) scoring, CTC decoding, and WER/CER reporting.

Parallels `harness/scores.py` for Task 2 (ASR across LibriSpeech clean/other,
TED-LIUM, and Multilingual LibriSpeech — MLS).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Standard Wav2Vec2 32-token English CTC vocabulary
DEFAULT_W2V2_CTC_VOCAB: Dict[int, str] = {
    0: "<pad>",
    1: "<s>",
    2: "</s>",
    3: "<unk>",
    4: "|",
    5: "E",
    6: "T",
    7: "A",
    8: "O",
    9: "N",
    10: "I",
    11: "H",
    12: "S",
    13: "R",
    14: "D",
    15: "L",
    16: "U",
    17: "M",
    18: "W",
    19: "C",
    20: "F",
    21: "G",
    22: "Y",
    23: "P",
    24: "B",
    25: "V",
    26: "K",
    27: "'",
    28: "X",
    29: "J",
    30: "Q",
    31: "Z",
}

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9' ]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_asr_text(text: str) -> str:
    """Normalize ASR transcript to uppercase A-Z, digits, apostrophes, and single spaces.

    Also strips TED-LIUM artifacts like `<unk>`, `ignore_time_segment_in_scoring`,
    and separated contraction tokens (` 's` -> `'s`).
    """
    s = str(text).upper()
    s = s.replace("<UNK>", " ").replace("IGNORE_TIME_SEGMENT_IN_SCORING", " ")
    # Join tokenized contractions common in TED-LIUM (e.g. "IT 'S" -> "IT'S", "DON 'T" -> "DON'T")
    s = re.sub(r"\s+'(S|T|RE|VE|M|LL|D)\b", r"'\1", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def edit_distance_tokens(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> int:
    """Exact Levenshtein edit distance between two token sequences."""
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        r_tok = ref_tokens[i - 1]
        for j in range(1, m + 1):
            cost = 0 if r_tok == hyp_tokens[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level Word Error Rate (WER = total word edits / total reference words)."""
    if len(references) != len(hypotheses):
        raise ValueError(f"references ({len(references)}) and hypotheses ({len(hypotheses)}) length mismatch")
    total_edits = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        r_words = normalize_asr_text(ref).split()
        h_words = normalize_asr_text(hyp).split()
        if not r_words:
            continue
        total_edits += edit_distance_tokens(r_words, h_words)
        total_words += len(r_words)
    if total_words == 0:
        return 0.0
    return float(total_edits / total_words)


def char_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level Character Error Rate (CER = total char edits / total reference chars)."""
    if len(references) != len(hypotheses):
        raise ValueError(f"references ({len(references)}) and hypotheses ({len(hypotheses)}) length mismatch")
    total_edits = 0
    total_chars = 0
    for ref, hyp in zip(references, hypotheses):
        r_chars = list(normalize_asr_text(ref))
        h_chars = list(normalize_asr_text(hyp))
        if not r_chars:
            continue
        total_edits += edit_distance_tokens(r_chars, h_chars)
        total_chars += len(r_chars)
    if total_chars == 0:
        return 0.0
    return float(total_edits / total_chars)


def ctc_greedy_decode(
    logits: np.ndarray,
    id_to_token: Optional[Dict[int, str]] = None,
    blank_id: int = 0,
    word_delimiter: str = "|",
    blank_bias_shift: float = 0.0,
) -> str:
    """Collapse frame-level CTC logits `(T, V)` into a normalized transcript string."""
    if id_to_token is None:
        id_to_token = DEFAULT_W2V2_CTC_VOCAB
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D logits (T, V), got {arr.shape}")
    if blank_bias_shift != 0.0:
        arr = arr.copy()
        arr[:, blank_id] += float(blank_bias_shift)
    pred_ids = np.argmax(arr, axis=-1).tolist()
    chars: List[str] = []
    prev_id = -1
    for idx in pred_ids:
        if idx != prev_id:
            if idx != blank_id:
                tok = id_to_token.get(int(idx), "")
                if tok == word_delimiter:
                    chars.append(" ")
                elif tok not in ("<pad>", "<s>", "</s>", "<unk>"):
                    chars.append(tok)
        prev_id = idx
    return normalize_asr_text("".join(chars))


def asr_report(rows: Sequence[dict]) -> dict:
    """Summarize corpus WER and CER overall, by dataset/split, and by acoustic condition."""
    if not rows:
        raise ValueError("no ASR rows provided")
    for row in rows:
        if "ref" not in row or "hyp" not in row:
            raise ValueError("each ASR row requires 'ref' and 'hyp'")

    def _block(subset: Sequence[dict]) -> dict:
        refs = [str(r["ref"]) for r in subset]
        hyps = [str(r["hyp"]) for r in subset]
        return {
            "n": len(subset),
            "wer": word_error_rate(refs, hyps),
            "cer": char_error_rate(refs, hyps),
        }

    overall = _block(rows)
    by_split: Dict[str, dict] = {}
    for split in sorted({str(r["split"]) for r in rows if r.get("split")}):
        by_split[split] = _block([r for r in rows if str(r.get("split")) == split])

    by_condition: Dict[str, dict] = {}
    for cond in sorted({str(r["condition"]) for r in rows if r.get("condition")}):
        by_condition[cond] = _block([r for r in rows if str(r.get("condition")) == cond])

    return {
        "n": overall["n"],
        "wer": overall["wer"],
        "cer": overall["cer"],
        "by_split": by_split,
        "by_condition": by_condition,
    }


def asr_counts_as_improvement(base: dict, edited: dict, tol: float = 0.005) -> dict:
    """An ASR edit counts when overall/noisy WER falls without regressing clean WER beyond `tol`."""
    reasons: List[str] = []
    if edited["wer"] >= base["wer"] - 1e-9:
        reasons.append("overall WER did not improve")
    for split, b_block in base.get("by_split", {}).items():
        e_block = edited.get("by_split", {}).get(split)
        if e_block and e_block["wer"] > b_block["wer"] + tol:
            reasons.append(f"split '{split}' WER regressed by more than {tol:.4f}")
    return {"counts": len(reasons) == 0, "reasons": reasons}
