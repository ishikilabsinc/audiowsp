import json

import numpy as np

from harness.sandbox import run_python


def _base(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    np.save(base / "encoder.weight.npy", np.ones((2, 2), dtype=np.float32))
    np.save(base / "head.bias.npy", np.arange(2, dtype=np.float32))
    return base


def test_save_edit_and_log(tmp_path):
    base = _base(tmp_path)
    work = tmp_path / "work"
    run = tmp_path / "run"
    run.mkdir()
    out = run_python(
        "w = load('encoder.weight').copy()\n"
        "w[0, 0] = w[0, 0] + 1\n"
        "print(save_edit('encoder.weight', w))\n"
        "print(edited())\n",
        work_dir=work,
        base_dir=base,
        run_dir=run,
        timeout=30,
    )
    assert "saved encoder.weight" in out
    saved = np.load(work / "edited" / "encoder.weight.npy")
    expected = np.ones((2, 2), dtype=np.float32)
    expected[0, 0] = 2
    assert np.allclose(saved, expected)
    lines = (run / "tool_calls.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "python"
    assert record["isolation"] == "process"
    assert record["returncode"] == 0
    state = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else {}
    # No state file was initialized, so the tool log does not invent one.
    assert state == {}


def test_rescale_and_frozen_are_refused(tmp_path):
    base = _base(tmp_path)
    out = run_python(
        "print(save_edit('encoder.weight', load_original('encoder.weight') * 4))\n",
        work_dir=tmp_path / "work-scale",
        base_dir=base,
        timeout=30,
    )
    assert "pure rescale" in out
    frozen = run_python(
        "print(save_edit('head.bias', load('head.bias') + 1))\n",
        work_dir=tmp_path / "work-frozen",
        base_dir=base,
        frozen=["head.bias"],
        timeout=30,
    )
    assert "frozen" in frozen
