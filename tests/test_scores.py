from pathlib import Path

from harness.scores import counts_as_improvement, read_jsonl, report

ROOT = Path(__file__).resolve().parents[1]


def _clean_rows():
    rows = []
    for split, generator in (("eval_matched", "A"), ("eval_unseen_generator", "B")):
        for speaker, gender, language, transcript in (
            ("s0", "m", "en", "t1"),
            ("s1", "m", "en", "t2"),
            ("s2", "f", "en", "t3"),
            ("s3", "f", "es", "t4"),
        ):
            rows.append(
                {
                    "score": 0.1,
                    "label": 0,
                    "generator": generator,
                    "speaker": speaker,
                    "gender": gender,
                    "language": language,
                    "transcript_id": transcript,
                    "split": split,
                }
            )
            rows.append(
                {
                    "score": 0.9,
                    "label": 1,
                    "generator": generator,
                    "speaker": speaker,
                    "gender": gender,
                    "language": language,
                    "transcript_id": transcript,
                    "split": split,
                }
            )
    return rows


def test_shortcut_example_is_not_general():
    summary = report(read_jsonl(ROOT / "examples" / "shortcut_scores.jsonl"))
    assert summary["by_split"]["eval_matched"]["eer"] == 0.0
    assert summary["by_split"]["eval_unseen_generator"]["eer"] > 0.2
    assert summary["gender_probe_auc_bona_fide"] == 1.0
    assert summary["speaker_dependence_bona_fide"] > 0.2
    assert summary["same_transcript_eer"] is not None
    assert summary["language_probe_positive"] == "en"


def test_clean_edit_counts():
    leaky = report(read_jsonl(ROOT / "examples" / "shortcut_scores.jsonl"))
    clean = report(_clean_rows())
    decision = counts_as_improvement(leaky, clean)
    assert decision["counts"] is True
    assert decision["reasons"] == []
    assert clean["by_split"]["eval_unseen_generator"]["eer"] == 0.0
    assert clean["gender_probe_auc_bona_fide"] == 0.5


def test_worse_shortcut_does_not_count():
    clean = report(_clean_rows())
    leaky = report(read_jsonl(ROOT / "examples" / "shortcut_scores.jsonl"))
    decision = counts_as_improvement(clean, leaky)
    assert decision["counts"] is False
    assert any("unseen-generator" in reason for reason in decision["reasons"])
