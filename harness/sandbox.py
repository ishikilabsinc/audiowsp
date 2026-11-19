"""Python kernel over a detector's weights.

This is process isolation: a fresh interpreter, a cleared environment, a time
limit, and a truncated reply. It is not an OS sandbox. bubblewrap is the
isolation the reference logs used, and this Mac does not have it. Do not point
an untrusted editor at this runner until that gap is closed.

save_edit refuses a frozen tensor and a pure rescale of a whole tensor.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from harness.edits import arch_from_base
from harness.firewall import assert_practice_path
from harness.log import log_tool

TIMEOUT = 900
MAX_OUT = 10_000

PRELUDE = r'''
import json, os
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=120)
WORK = os.environ["WSP_WORK"]
BASE = os.environ["WSP_BASE"]
_ARCH = json.load(open(os.path.join(WORK, "arch.json")))
NAMES = list(_ARCH["tensors"])
FROZEN = set(_ARCH.get("frozen", []))
SHAPES = {key: tuple(value) for key, value in _ARCH["tensors"].items()}

def _path(root, name):
    return os.path.join(root, name + ".npy")

def load_original(name):
    if name not in SHAPES:
        raise KeyError(name)
    return np.load(_path(BASE, name)).astype(np.float32)

def load(name):
    edited = _path(os.path.join(WORK, "edited"), name)
    if os.path.exists(edited):
        return np.load(edited).astype(np.float32)
    return load_original(name)

def _pure_rescale(new, old):
    old = np.asarray(old, np.float64).ravel()
    new = np.asarray(new, np.float64).ravel()
    denom = float(np.dot(old, old))
    if denom == 0.0:
        return False
    scale = float(np.dot(new, old) / denom)
    residual = np.linalg.norm(new - scale * old) / np.linalg.norm(old)
    return residual < 1e-3 and abs(scale - 1.0) > 1e-6

def save_edit(name, array):
    if name in FROZEN:
        raise ValueError("refused: %s is frozen" % name)
    array = np.asarray(array, np.float32)
    if tuple(array.shape) != SHAPES[name]:
        raise ValueError("shape %s != %s for %s" % (array.shape, SHAPES[name], name))
    if not np.isfinite(array).all():
        raise ValueError("%s has non-finite values" % name)
    if _pure_rescale(array, load_original(name)):
        raise ValueError("refused: %s is a pure rescale of the original tensor" % name)
    os.makedirs(os.path.join(WORK, "edited"), exist_ok=True)
    np.save(_path(os.path.join(WORK, "edited"), name), array)
    return "saved %s shape=%s" % (name, tuple(array.shape))

def drop_edit(name):
    path = _path(os.path.join(WORK, "edited"), name)
    if os.path.exists(path):
        os.remove(path)
    return name

def edited():
    directory = os.path.join(WORK, "edited")
    if not os.path.isdir(directory):
        return []
    return sorted(name[:-4] for name in os.listdir(directory) if name.endswith(".npy"))

def ofp_solve_numpy(X, y, nuis_deltas=None, lam=25.0):
    """In-sandbox NumPy One-Forward-Pass (OFP) Nuisance-Orthogonal Woodbury solver."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    pos = X[y > 0.5]
    neg = X[y <= 0.5]
    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    Xc = np.vstack([pos - mu_pos, neg - mu_neg])
    cov = (Xc.T @ Xc) / max(len(X), 1)
    if nuis_deltas is not None:
        dN = np.asarray(nuis_deltas, np.float64)
        cov = cov + 4.0 * (dN.T @ dN) / max(len(dN), 1)
    cov += float(lam) * np.eye(X.shape[1], dtype=np.float64)
    w = np.linalg.solve(cov, mu_pos - mu_neg)
    b = -0.5 * float(np.dot(mu_pos + mu_neg, w))
    return w.astype(np.float32), np.float32(b)
'''


def isolation_mode() -> str:
    return "process"


def run_python(
    code: str,
    *,
    work_dir: Path,
    base_dir: Path,
    run_dir: Path | None = None,
    frozen: list[str] | None = None,
    timeout: int = TIMEOUT,
) -> str:
    work_dir = assert_practice_path(Path(work_dir), context="sandbox.work_dir")
    base_dir = assert_practice_path(Path(base_dir), context="sandbox.base_dir")
    work_dir.mkdir(parents=True, exist_ok=True)
    arch = arch_from_base(base_dir, frozen)
    (work_dir / "arch.json").write_text(json.dumps(arch))
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(work_dir),
        "WSP_WORK": str(work_dir),
        "WSP_BASE": str(base_dir),
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "LC_ALL": "C",
    }
    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", PRELUDE + "\n" + code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(work_dir),
        )
        out = completed.stdout
        if completed.stderr.strip():
            out += "\n[stderr]\n" + completed.stderr[-6000:]
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        out = f"[timed out after {timeout} s]"
        returncode = -1
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    out = out or "(no output)"
    if run_dir is not None:
        log_tool(
            run_dir,
            kind="python",
            code=code,
            out=out,
            seconds=time.time() - started,
            isolation=isolation_mode(),
            returncode=returncode,
        )
    return out
