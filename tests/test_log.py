import json
from pathlib import Path

import pytest

from harness.log import PromptNotReady, log_turn, placeholders_left, seed_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_seed_records_the_draft_once(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "instructions.md").write_text("Answer the user's request.\n")
    (exp / "message.txt").write_text("Look at {checkpoint_name} with budget {forward_budget}.\n")
    first = seed_experiment(exp)
    second = seed_experiment(exp)
    assert first == second == 1
    prompt = (exp / "runs" / "p00_scaffold" / "turns" / "t001" / "prompt.txt").read_text()
    assert prompt == (exp / "message.txt").read_text()
    record = json.loads((exp / "runs" / "p00_scaffold" / "turns" / "t001" / "run.json").read_text())
    assert record["sent"] is False
    assert "{checkpoint_name}" in record["placeholders"]
    state = json.loads((exp / "runs" / "p00_scaffold" / "state.json").read_text())
    assert state["draft_turns"] == 1
    assert state["sent_turns"] == 0


def test_sent_turn_rejects_placeholders(tmp_path):
    with pytest.raises(PromptNotReady):
        log_turn(tmp_path, "edit {checkpoint_name}\n", sent=True)
    assert placeholders_left("budget {forward_budget} and {practice_manifest}") == [
        "{forward_budget}",
        "{practice_manifest}",
    ]


def test_sent_turn_is_recorded(tmp_path):
    number = log_turn(tmp_path, "The checkpoint is bound.\n", sent=True, model="editor", effort="max")
    assert number == 1
    record = json.loads((tmp_path / "turns" / "t001" / "run.json").read_text())
    assert record["sent"] is True
    assert record["model"] == "editor"
    assert record["placeholders"] == []
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["sent_turns"] == 1
    assert state["editor"] == "editor"


def test_committed_prompt_matches_message():
    message = (ROOT / "exp1_unseen_generators" / "message.txt").read_text()
    prompt = (
        ROOT / "exp1_unseen_generators" / "runs" / "p00_scaffold" / "turns" / "t001" / "prompt.txt"
    ).read_text()
    assert prompt == message
    record = json.loads(
        (ROOT / "exp1_unseen_generators" / "runs" / "p00_scaffold" / "turns" / "t001" / "run.json").read_text()
    )
    assert record["sent"] is False
    assert set(record["placeholders"]) == {
        "{checkpoint_name}",
        "{forward_budget}",
        "{practice_manifest}",
    }
