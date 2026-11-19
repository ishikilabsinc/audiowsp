import json

import pytest

from harness.forward import ForwardNotReady, run_forward


def test_forward_refuses_without_a_checkpoint():
    with pytest.raises(ForwardNotReady):
        run_forward({"type": "score", "model": "original", "items": ["a"]}, None)


def test_forward_refuses_a_named_checkpoint_until_audio_is_wired():
    with pytest.raises(ForwardNotReady, match="not implemented"):
        run_forward({"type": "activations", "model": "edited", "items": ["a"], "layers": [0]}, "some-detector")


def test_doctor_reports_process_isolation():
    from harness.doctor import checks

    info = checks()
    assert info["python_ok"] is True
    assert info["numpy"]
    assert info["editor_called"] is False
    assert info["checkpoint"] is None
    assert info["isolation"] in ("process", "bwrap")


def test_forward_score_and_activations_with_checkpoint(tmp_path):
    import numpy as np
    from harness.models import init_checkpoint

    base = init_checkpoint(tmp_path / "base", mode="scratch")
    work = tmp_path / "work"
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"id": "u0", "label": 0, "split": "eval_matched", "generator": "bonafide", "speaker": "s1", "gender": "f", "language": "en", "transcript_id": "t1"},
        {"id": "u1", "label": 1, "split": "eval_matched", "generator": "A07", "speaker": "s1", "gender": "f", "language": "en", "transcript_id": "t1"},
        {"id": "u2", "label": 0, "split": "eval_unseen_generator", "generator": "bonafide", "speaker": "s2", "gender": "m", "language": "zh", "transcript_id": "t2"},
        {"id": "u3", "label": 1, "split": "eval_unseen_generator", "generator": "ASV5_A24", "speaker": "s2", "gender": "m", "language": "zh", "transcript_id": "t2"},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cache_path = tmp_path / "cache.npz"
    np.savez(
        cache_path,
        ids=np.array(["u0", "u1", "u2", "u3"]),
        layer0=np.zeros((4, 1024), dtype=np.float32),
        layer1=np.zeros((4, 1024), dtype=np.float32),
        layer2=np.zeros((4, 1024), dtype=np.float32),
        layer3=np.zeros((4, 160), dtype=np.float32),
        layer4=np.zeros((4, 128), dtype=np.float32),
        w2v2_score=np.array([-5.0, 4.0, -4.0, 5.0], dtype=np.float32),
    )
    act_res = run_forward(
        {"type": "activations", "model": "original", "items": ["u0", "u2"], "layers": [0, 3]},
        str(base),
        base_dir=base,
        work_dir=work,
        manifest_path=manifest,
        cache_path=cache_path,
    )
    assert act_res["n_items"] == 2
    assert (work / "act_layer0.npy").exists()
    score_res = run_forward(
        {"type": "score", "model": "both", "items": ["u0", "u1", "u2", "u3"]},
        str(base),
        base_dir=base,
        work_dir=work,
        manifest_path=manifest,
        cache_path=cache_path,
    )
    assert score_res["n_items"] == 4
    assert "decision" in score_res

