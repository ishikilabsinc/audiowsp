"""Prompt log laid out like the runs in RikiyaT/lm_understand-logs.

Every message intended for the editor model is written under turns/tNNN/
before anyone sends it. A turn with sent=false is a draft.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

# Filled in by the harness once a checkpoint, a budget, and a practice
# manifest exist. log_turn(sent=True) refuses a prompt that still has one.
PLACEHOLDERS = ("{checkpoint_name}", "{forward_budget}", "{practice_manifest}")


class PromptNotReady(Exception):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("prompt still has placeholders: " + ", ".join(missing))


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def placeholders_left(text: str) -> list[str]:
    return [item for item in PLACEHOLDERS if item in text]


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def turn_dir(run_dir: Path, number: int) -> Path:
    return run_dir / "turns" / f"t{number:03d}"


def list_turns(run_dir: Path) -> list[int]:
    turns = Path(run_dir) / "turns"
    if not turns.is_dir():
        return []
    numbers = []
    for path in turns.iterdir():
        if path.is_dir() and path.name.startswith("t") and path.name[1:].isdigit():
            numbers.append(int(path.name[1:]))
    return sorted(numbers)


def read_state(run_dir: Path) -> dict:
    path = Path(run_dir) / "state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_state(run_dir: Path, state: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def init_run(
    run_dir: Path,
    *,
    budget: int | None,
    mode: str,
    prompt_version: str,
    instructions: str,
) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    instructions_path = run_dir / "instructions.md"
    if not instructions_path.exists():
        instructions_path.write_text(_newline(instructions))
    state = read_state(run_dir)
    if not state:
        state = {
            "budget": budget,
            "mode": mode,
            "prompt": prompt_version,
            "turn": 0,
            "runs": 0,
            "passes": 0,
            "done": False,
            "sent_turns": 0,
            "draft_turns": 0,
            "editor": None,
            "notes": [],
        }
        write_state(run_dir, state)
    return state


def log_turn(
    run_dir: Path,
    prompt: str,
    *,
    answer: str | None = None,
    sent: bool = False,
    model: str | None = None,
    effort: str | None = None,
    note: str = "",
) -> int:
    """Record one editor message. Refuses sent=True while placeholders remain."""
    run_dir = Path(run_dir)
    if sent:
        missing = placeholders_left(prompt)
        if missing:
            raise PromptNotReady(missing)
    numbers = list_turns(run_dir)
    number = (numbers[-1] + 1) if numbers else 1
    dest = turn_dir(run_dir, number)
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "prompt.txt").write_text(_newline(prompt))
    if answer is not None:
        (dest / "answer.md").write_text(_newline(answer))
    (dest / "stderr.log").write_text("")
    digest = prompt_sha256(prompt)
    event = {
        "type": "turn.recorded",
        "sent": sent,
        "prompt_sha256": digest,
        "chars": len(prompt),
        "note": note,
    }
    (dest / "events.jsonl").write_text(json.dumps(event) + "\n")
    record = {
        "ok": True,
        "sent": sent,
        "reason": None if sent else "recorded locally; the editor model has not been called",
        "limit": False,
        "wall_s": 0.0,
        "model": model,
        "effort": effort,
        "audit": {
            "tool_free": True,
            "tool_items": {},
            "event_types": {"turn.recorded": 1},
        },
        "prompt_sha256": digest,
        "placeholders": placeholders_left(prompt),
        "end": _now(),
        "note": note,
    }
    (dest / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    state = read_state(run_dir)
    if not state:
        state = {
            "budget": None,
            "mode": "audio-deepfake",
            "prompt": "v0",
            "turn": 0,
            "runs": 0,
            "passes": 0,
            "done": False,
            "sent_turns": 0,
            "draft_turns": 0,
            "editor": None,
            "notes": [],
        }
    state["turn"] = number
    if sent:
        state["sent_turns"] = int(state.get("sent_turns", 0)) + 1
        state["editor"] = model
    else:
        state["draft_turns"] = int(state.get("draft_turns", 0)) + 1
    write_state(run_dir, state)
    _append_driver(
        run_dir,
        f"turn {number:03d} recorded sent={str(sent).lower()} sha256={digest[:12]} chars={len(prompt)}",
    )
    return number


def log_tool(
    run_dir: Path,
    *,
    kind: str,
    code: str,
    out: str,
    seconds: float,
    isolation: str,
    returncode: int,
) -> None:
    run_dir = Path(run_dir)
    line = {
        "t": datetime.now().astimezone().timestamp(),
        "kind": kind,
        "s": round(seconds, 2),
        "isolation": isolation,
        "returncode": returncode,
        "code": code,
        "out": out,
    }
    with (run_dir / "tool_calls.jsonl").open("a") as handle:
        handle.write(json.dumps(line) + "\n")
    state = read_state(run_dir)
    if state:
        if kind == "python":
            state["runs"] = int(state.get("runs", 0)) + 1
        elif kind == "forward":
            state["passes"] = int(state.get("passes", 0)) + 1
        write_state(run_dir, state)


def _append_driver(run_dir: Path, text: str) -> None:
    line = f"{_now()} {text}\n"
    for name in ("driver.log", "loop.log"):
        with (Path(run_dir) / name).open("a") as handle:
            handle.write(line)


def seed_experiment(exp_dir: Path) -> int:
    """Record the experiment's message.txt as a draft turn, once."""
    exp_dir = Path(exp_dir)
    run_dir = exp_dir / "runs" / "p00_scaffold"
    instructions = (exp_dir / "instructions.md").read_text()
    message = (exp_dir / "message.txt").read_text()
    init_run(
        run_dir,
        budget=None,
        mode="audio-deepfake",
        prompt_version="v0",
        instructions=instructions,
    )
    existing = list_turns(run_dir)
    if existing:
        return existing[-1]
    return log_turn(
        run_dir,
        message,
        sent=False,
        note="draft task; placeholders are unbound and the editor has not been called",
    )
