import numpy as np

from harness.edits import apply_diff, is_pure_rescale, make_diff, sha256_array


def test_pure_rescale():
    base = np.arange(6, dtype=np.float32).reshape(2, 3) + 1
    assert is_pure_rescale(base * 3, base)
    assert not is_pure_rescale(base, base)
    changed = base.copy()
    changed[0, 0] += 1
    assert not is_pure_rescale(changed, base)


def test_diff_roundtrip(tmp_path):
    base = tmp_path / "base"
    edited = tmp_path / "edited"
    diff = tmp_path / "diff"
    rebuilt = tmp_path / "rebuilt"
    base.mkdir()
    edited.mkdir()
    weight = np.arange(16, dtype=np.float32).reshape(4, 4)
    bias = np.zeros(4, dtype=np.float32)
    np.save(base / "encoder.weight.npy", weight)
    np.save(base / "head.bias.npy", bias)
    changed = weight.copy()
    changed[1, 2] = 7.5
    np.save(edited / "encoder.weight.npy", changed)
    manifest = make_diff(base, edited, diff)
    assert manifest["changed_tensors"] == 1
    assert manifest["tensors"][0]["name"] == "encoder.weight"
    assert manifest["tensors"][0]["changed_entries"] == 1
    assert manifest["tensors"][0]["changed_rows"] == 1
    assert manifest["tensors"][0]["changed_columns"] == 1
    out = apply_diff(base, diff, rebuilt)
    assert sha256_array(out["encoder.weight"]) == manifest["tensors"][0]["edited_sha256"]
    assert out["encoder.weight"][1, 2] == np.float32(7.5)
