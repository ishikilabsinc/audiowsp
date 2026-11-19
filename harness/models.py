"""AudioTruFor and W2V2-AASIST-TruFor weight-space programmable architectures.

All weights live as explicit float32 .npy tensors in a checkpoint directory so
the sandbox can inspect and write them with load() and save_edit() without
gradient descent or backpropagation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

BRANCH_SPECS = (
    ("b0", "layer0", 1024),  # log temporal std of SSL frames
    ("b1", "layer1", 1024),  # log temporal std of 1st-order frame difference
    ("bm", "layer2", 1024),  # temporal mean of SSL frames
    ("bh", "layer3", 160),   # AASIST spectro-temporal graph readout vector
)


def init_checkpoint(
    base_dir: Path,
    *,
    aasist_dir: Path | None = None,
    mode: str = "w2v2",
) -> Path:
    """Create a base weight directory for 'w2v2' or 'scratch' mode."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    if aasist_dir is not None and Path(aasist_dir).is_dir():
        for npy in sorted(Path(aasist_dir).glob("*.npy")):
            shutil.copy2(npy, base_dir / npy.name)
    if not (base_dir / "out_layer.weight.npy").exists():
        np.save(base_dir / "out_layer.weight.npy", np.zeros((2, 160), dtype=np.float32))
    if not (base_dir / "out_layer.bias.npy").exists():
        np.save(base_dir / "out_layer.bias.npy", np.zeros((2,), dtype=np.float32))

    temp_w = np.zeros((2, 1, 3), dtype=np.float32)
    temp_w[0, 0, 1] = 1.0
    temp_w[1, 0, 1] = 1.0
    temp_w[1, 0, 2] = -1.0
    np.save(base_dir / "trufor.temp_conv.W.npy", temp_w)

    for bname, _layer_key, dim in BRANCH_SPECS:
        np.save(base_dir / f"trufor.{bname}.shift.npy", np.zeros((dim,), dtype=np.float32))
        np.save(base_dir / f"trufor.{bname}.scale.npy", np.ones((dim,), dtype=np.float32))
        if mode == "scratch":
            np.save(base_dir / f"trufor.{bname}.proj.npy", np.eye(dim, dtype=np.float32))
            np.save(
                base_dir / f"trufor.{bname}.w.npy",
                np.full((dim,), 1.0 / dim, dtype=np.float32),
            )
        else:
            np.save(base_dir / f"trufor.{bname}.proj.npy", np.zeros((dim, dim), dtype=np.float32))
            np.save(base_dir / f"trufor.{bname}.w.npy", np.zeros((dim,), dtype=np.float32))
        np.save(
            base_dir / f"trufor.{bname}.norm.npy",
            np.array([0.0, 1.0], dtype=np.float32),
        )

    np.save(base_dir / "trufor.ch.shift.npy", np.zeros((128,), dtype=np.float32))
    np.save(base_dir / "trufor.ch.scale.npy", np.ones((128,), dtype=np.float32))
    np.save(base_dir / "trufor.ch.w.npy", np.zeros((128,), dtype=np.float32))

    if mode == "scratch":
        alpha = np.array([0.40, 0.20, 0.20, 0.20, 0.0, 0.0], dtype=np.float32)
    else:
        alpha = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.save(base_dir / "trufor.head.alpha.npy", alpha)
    return base_dir


def load_weight_set(base_dir: Path, edited_dir: Path | None = None) -> tuple[dict[str, np.ndarray], set[str]]:
    """Load float32 tensors from base_dir overlaid with edited_dir."""
    base_dir = Path(base_dir)
    weights: dict[str, np.ndarray] = {}
    edited_names: set[str] = set()
    for path in sorted(base_dir.glob("*.npy")):
        name = path.name[:-4]
        weights[name] = np.load(path).astype(np.float32)
    if edited_dir is not None and Path(edited_dir).is_dir():
        for path in sorted(Path(edited_dir).glob("*.npy")):
            name = path.name[:-4]
            arr = np.load(path).astype(np.float32)
            if name not in weights or not np.array_equal(arr, weights[name]):
                edited_names.add(name)
            weights[name] = arr
    return weights, edited_names


def forward_scores(
    cache: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    edited_names: set[str],
    indices: np.ndarray,
) -> np.ndarray:
    """Execute the deterministic forward pass over cached H100 representations."""
    alpha = weights["trufor.head.alpha"].astype(np.float64)
    total = np.zeros(len(indices), dtype=np.float64)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for idx, (bname, layer_key, _dim) in enumerate(BRANCH_SPECS):
            w_branch = float(alpha[idx])
            if w_branch == 0.0:
                continue
            x = cache[layer_key][indices].astype(np.float64)
            shift = weights[f"trufor.{bname}.shift"].astype(np.float64)
            scale = weights[f"trufor.{bname}.scale"].astype(np.float64)
            proj = weights[f"trufor.{bname}.proj"].astype(np.float64)
            w = weights[f"trufor.{bname}.w"].astype(np.float64)
            norm = weights[f"trufor.{bname}.norm"].astype(np.float64)

            z = (x + shift) * scale
            r = z @ proj
            a = np.log1p(r * r)
            s = (a @ w - norm[0]) * norm[1]
            total += w_branch * s

        w_ch = float(alpha[4])
        if w_ch != 0.0:
            x_ch = cache["layer4"][indices].astype(np.float64)
            ch_shift = weights["trufor.ch.shift"].astype(np.float64)
            ch_scale = weights["trufor.ch.scale"].astype(np.float64)
            ch_w = weights["trufor.ch.w"].astype(np.float64)
            s_ch = ((x_ch + ch_shift) * ch_scale) @ ch_w
            total += w_ch * s_ch

        w_aasist = float(alpha[5])
        if w_aasist != 0.0:
            if "out_layer.weight" in edited_names or "out_layer.bias" in edited_names:
                h = cache["layer3"][indices].astype(np.float64)
                w_out = weights["out_layer.weight"].astype(np.float64)
                b_out = weights["out_layer.bias"].astype(np.float64)
                s_aasist = -(h @ w_out[1] + b_out[1])
            else:
                s_aasist = cache["w2v2_score"][indices].astype(np.float64)
            total += w_aasist * s_aasist

    if not np.isfinite(total).all():
        raise ValueError("forward_scores produced non-finite values")
    return total.astype(np.float32)
