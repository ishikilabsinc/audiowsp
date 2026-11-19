"""Report whether this machine can run the scaffold."""

from __future__ import annotations

import shutil
import sys


def checks() -> dict:
    numpy_version = None
    numpy_error = None
    try:
        import numpy

        numpy_version = numpy.__version__
    except Exception as exc:  # pragma: no cover - depends on the environment
        numpy_error = str(exc)
    bwrap = shutil.which("bwrap") is not None
    return {
        "python": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 9),
        "numpy": numpy_version,
        "numpy_error": numpy_error,
        "bwrap": bwrap,
        "isolation": "bwrap" if bwrap else "process",
        "editor_called": False,
        "checkpoint": None,
    }


def render() -> str:
    info = checks()
    lines = [
        f"python {info['python']} ({'ok' if info['python_ok'] else 'need 3.9+'})",
        f"numpy {info['numpy'] or 'missing'}",
        f"sandbox isolation: {info['isolation']}",
        "editor model: not called",
        "detector checkpoint: not configured",
    ]
    if not info["bwrap"]:
        lines.append(
            "bubblewrap is not installed. The Python runner is a fresh process with a "
            "time limit, not an OS sandbox. Install bubblewrap before an editor session "
            "is allowed to run code."
        )
    return "\n".join(lines)
