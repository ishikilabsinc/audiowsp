"""Forward-pass tool for scoring and reading activations under a pass budget.

Supports:
  - {"type": "score", "model": "original" | "edited" | "both", "items": [...]}
  - {"type": "activations", "model": "original" | "edited", "items": [...], "layers": [0, ...]}
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from harness.firewall import assert_practice_manifest_rows, assert_practice_path
from harness.log import log_tool, read_state, write_state
from harness.models import forward_scores, load_weight_set
from harness.scores import counts_as_improvement, read_jsonl, report

MAX_SCORE_ITEMS = 512
MAX_ACT_ITEMS = 128
VALID_LAYERS = {0, 1, 2, 3, 4}


class ForwardNotReady(Exception):
    pass


def _resolve_checkpoint_dir(checkpoint: str | Path | None, base_dir: Path | None) -> Path:
    if base_dir is not None and Path(base_dir).is_dir():
        return Path(base_dir)
    if not checkpoint:
        raise ForwardNotReady(
            "no detector checkpoint is configured; refusing to score or read activations"
        )
    cand = Path(checkpoint)
    if cand.is_dir() and any(cand.glob("*.npy")):
        return cand
    raise ForwardNotReady(
        f"checkpoint {checkpoint} is named, but the audio forward pass is not implemented yet"
    )


def _load_manifest_and_cache(
    manifest_path: Path | None,
    cache_path: Path | None,
) -> tuple[dict[str, dict], dict[str, int], dict[str, np.ndarray]]:
    m_path = Path(
        manifest_path
        or os.environ.get(
            "WSP_MANIFEST",
            "/Users/amityadav/Documents/weight-space-programming-for-audio-models/exp1_unseen_generators/manifests/practice_manifest.jsonl",
        )
    )
    c_path = Path(
        cache_path
        or os.environ.get("WSP_CACHE", "/tmp/asv_data/dataset/practice_h100_cache.npz")
    )
    if not m_path.exists() or not c_path.exists():
        raise ForwardNotReady(f"missing manifest ({m_path}) or feature cache ({c_path})")
    assert_practice_path(m_path, context="forward.manifest_path")
    assert_practice_path(c_path, context="forward.cache_path")
    rows = read_jsonl(m_path)
    assert_practice_manifest_rows(rows, context="forward.manifest")
    by_id = {str(r["id"]): r for r in rows}
    cache_npz = np.load(c_path, allow_pickle=False)
    cache = {k: cache_npz[k] for k in cache_npz.files}
    id_to_idx = {str(uid): idx for idx, uid in enumerate(cache["ids"].tolist())}
    return by_id, id_to_idx, cache


def run_forward(
    request: dict,
    checkpoint: str | None,
    *,
    base_dir: Path | None = None,
    work_dir: Path | None = None,
    run_dir: Path | None = None,
    manifest_path: Path | None = None,
    cache_path: Path | None = None,
) -> dict:
    if not checkpoint and base_dir is None:
        raise ForwardNotReady(
            "no detector checkpoint is configured; refusing to score or read activations"
        )
    kind = request.get("type")
    if kind not in ("score", "activations"):
        raise ValueError("forward type must be 'score' or 'activations'")

    ckpt_dir = _resolve_checkpoint_dir(checkpoint, base_dir)
    if run_dir is not None:
        state = read_state(Path(run_dir))
        budget = state.get("budget")
        passes = int(state.get("passes", 0))
        if budget is not None and passes >= int(budget):
            raise RuntimeError(f"forward budget exhausted ({passes}/{budget})")

    by_id, id_to_idx, cache = _load_manifest_and_cache(manifest_path, cache_path)
    items = request.get("items")
    if items == "all":
        items = list(by_id.keys())
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a non-empty list of clip ids")

    missing = [item for item in items if str(item) not in by_id or str(item) not in id_to_idx]
    if missing:
        raise KeyError(f"unknown item ids: {missing[:5]}")

    started = time.time()
    indices = np.array([id_to_idx[str(item)] for item in items], dtype=np.int64)
    edited_dir = (Path(work_dir) / "edited") if work_dir is not None else None

    if kind == "score":
        if len(items) > MAX_SCORE_ITEMS and run_dir is not None:
            raise ValueError(f"score pass allows at most {MAX_SCORE_ITEMS} items, got {len(items)}")
        model_choice = request.get("model", "both")
        if model_choice not in ("original", "edited", "both"):
            raise ValueError("model must be 'original', 'edited', or 'both'")

        result: dict = {"type": "score", "model": model_choice, "n_items": len(items)}
        if model_choice in ("original", "both"):
            w_orig, ed_orig = load_weight_set(ckpt_dir, None)
            s_orig = forward_scores(cache, w_orig, ed_orig, indices)
            rows_orig = [{**by_id[str(item)], "score": float(sc)} for item, sc in zip(items, s_orig)]
            result["original"] = report(rows_orig)
        if model_choice in ("edited", "both"):
            w_edit, ed_edit = load_weight_set(ckpt_dir, edited_dir)
            s_edit = forward_scores(cache, w_edit, ed_edit, indices)
            rows_edit = [{**by_id[str(item)], "score": float(sc)} for item, sc in zip(items, s_edit)]
            result["edited"] = report(rows_edit)
        if model_choice == "both":
            orig_rep = result["original"]
            edit_rep = result["edited"]
            result["paired_eer_diff"] = float(edit_rep["eer"] - orig_rep["eer"])
            orig_unseen = orig_rep.get("by_split", {}).get("eval_unseen_generator", {}).get("eer")
            edit_unseen = edit_rep.get("by_split", {}).get("eval_unseen_generator", {}).get("eer")
            if orig_unseen is not None and edit_unseen is not None:
                result["paired_unseen_eer_diff"] = float(edit_unseen - orig_unseen)
            result["decision"] = counts_as_improvement(orig_rep, edit_rep)
    else:
        if len(items) > MAX_ACT_ITEMS:
            raise ValueError(f"activations pass allows at most {MAX_ACT_ITEMS} items, got {len(items)}")
        layers = request.get("layers", [])
        if not isinstance(layers, list) or not layers or any(int(l) not in VALID_LAYERS for l in layers):
            raise ValueError(f"layers must be a non-empty subset of {sorted(VALID_LAYERS)}")
        if work_dir is None:
            raise ValueError("activations pass requires work_dir")
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        written = {}
        for l_idx in layers:
            l_int = int(l_idx)
            arr = cache[f"layer{l_int}"][indices].astype(np.float32)
            out_file = work_path / f"act_layer{l_int}.npy"
            np.save(out_file, arr)
            written[f"layer{l_int}"] = {"path": str(out_file), "shape": list(arr.shape)}
        (work_path / "act_items.json").write_text(json.dumps([str(i) for i in items], indent=2) + "\n")
        result = {
            "type": "activations",
            "model": request.get("model", "original"),
            "n_items": len(items),
            "written": written,
        }

    if run_dir is not None:
        log_tool(
            Path(run_dir),
            kind="forward",
            code=json.dumps(request),
            out=json.dumps(result),
            seconds=time.time() - started,
            isolation="process",
            returncode=0,
        )
    return result
