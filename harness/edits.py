"""Exact float32 weight diffs.

A run stores the tensors it changed. make_diff writes a sparse npz plus a
manifest of sha256 hashes. apply_diff rebuilds those tensors from the base
weights and checks every hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def is_pure_rescale(new: np.ndarray, old: np.ndarray, tol: float = 1e-3) -> bool:
    """True when new is a scalar multiple of old and the scalar is not 1."""
    old_flat = np.asarray(old, dtype=np.float64).ravel()
    new_flat = np.asarray(new, dtype=np.float64).ravel()
    if old_flat.shape != new_flat.shape:
        return False
    denom = float(np.dot(old_flat, old_flat))
    if denom == 0.0:
        return False
    scale = float(np.dot(new_flat, old_flat) / denom)
    residual = float(np.linalg.norm(new_flat - scale * old_flat) / np.linalg.norm(old_flat))
    return residual < tol and abs(scale - 1.0) > 1e-6


def list_tensors(directory: Path) -> dict[str, np.ndarray]:
    directory = Path(directory)
    found = {}
    for path in sorted(directory.glob("*.npy")):
        found[path.name[: -len(".npy")]] = np.load(path)
    return found


def arch_from_base(base_dir: Path, frozen: list[str] | None = None) -> dict:
    tensors = {}
    for name, array in list_tensors(base_dir).items():
        tensors[name] = list(array.shape)
    return {"tensors": tensors, "frozen": list(frozen or [])}


def make_diff(base_dir: Path, edited_dir: Path, out_dir: Path) -> dict:
    base = list_tensors(base_dir)
    edited = list_tensors(edited_dir)
    unknown = sorted(set(edited) - set(base))
    if unknown:
        raise ValueError("edited tensors missing from the base checkpoint: " + ", ".join(unknown))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    records = []
    for name in sorted(edited):
        base_array = np.ascontiguousarray(base[name], dtype=np.float32)
        edited_array = np.ascontiguousarray(edited[name], dtype=np.float32)
        if base_array.shape != edited_array.shape:
            raise ValueError(f"{name}: shape {edited_array.shape} != {base_array.shape}")
        changed = base_array != edited_array
        if not changed.any():
            continue
        index = np.flatnonzero(changed).astype(np.int64)
        slot = len(records)
        arrays[f"t{slot}_idx"] = index
        arrays[f"t{slot}_val"] = edited_array.reshape(-1)[index]
        base_norm = float(np.linalg.norm(base_array))
        relative = float(np.linalg.norm(edited_array - base_array) / base_norm) if base_norm else 0.0
        record = {
            "name": name,
            "shape": list(base_array.shape),
            "dtype": "float32",
            "form": "sparse",
            "changed_entries": int(changed.sum()),
            "relative_change": relative,
            "base_sha256": sha256_array(base_array),
            "edited_sha256": sha256_array(edited_array),
        }
        if base_array.ndim == 2:
            record["changed_rows"] = int(changed.any(axis=1).sum())
            record["changed_columns"] = int(changed.any(axis=0).sum())
        records.append(record)
    np.savez(out_dir / "edit.npz", **arrays)
    manifest = {
        "dtype": "float32",
        "tensors": records,
        "changed_tensors": len(records),
        "changed_entries_total": int(sum(item["changed_entries"] for item in records)),
        "note": "sha256 is of the contiguous float32 bytes",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def apply_diff(base_dir: Path, diff_dir: Path, out_dir: Path) -> dict[str, np.ndarray]:
    """Rebuild edited tensors and check both hashes in the manifest."""
    diff_dir = Path(diff_dir)
    manifest = json.loads((diff_dir / "manifest.json").read_text())
    packed = np.load(diff_dir / "edit.npz")
    base = list_tensors(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rebuilt = {}
    for slot, record in enumerate(manifest["tensors"]):
        name = record["name"]
        array = np.array(base[name], dtype=np.float32, copy=True)
        if sha256_array(np.ascontiguousarray(array)) != record["base_sha256"]:
            raise ValueError(f"{name}: base sha256 does not match the manifest")
        flat = array.reshape(-1)
        flat[packed[f"t{slot}_idx"]] = packed[f"t{slot}_val"]
        edited = np.ascontiguousarray(array, dtype=np.float32)
        digest = sha256_array(edited)
        if digest != record["edited_sha256"]:
            raise ValueError(f"{name}: rebuilt sha256 {digest} != {record['edited_sha256']}")
        np.save(out_dir / f"{name}.npy", edited)
        rebuilt[name] = edited
    return rebuilt
