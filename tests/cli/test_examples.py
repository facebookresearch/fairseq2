import io
import sys
from pathlib import Path

import pytest

import fairseq2.cli.commands

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = [s.name for s in EXAMPLE_DIR.glob("*.py") if "data" not in s.name]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script(script: str) -> None:
    try:
        fairseq2.cli.commands.help(EXAMPLE_DIR / script)
    except ImportError as e:
        pytest.skip(f"Example requires external dependencies: {e}")


def test_train_simple_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Training
    steps = 100
    simple_task = Path(__file__).parent / "simple_task.py"
    train_metrics = fairseq2.cli.commands.train(
        simple_task, workdir=tmp_path, max_steps=steps, num_gpus=0
    )
    assert "train/lr" in train_metrics

    # Evaluation
    snapshot = next(tmp_path.glob("*/epoch_2_step_100.train"), None)
    assert snapshot
    eval_metrics = fairseq2.cli.commands.evaluate(snapshot, num_gpus=0)
    assert "eval/loss" in eval_metrics

    # Inference
    with monkeypatch.context() as m:
        m.setattr(sys, "stdin", io.StringIO("40 2\n25 25\n"))
        m.setattr(sys.stdin, "fileno", lambda: 128)
        stdout = io.StringIO()
        m.setattr(sys, "stdout", stdout)
        fairseq2.cli.commands.inference(snapshot, num_gpus=0)

    assert len(stdout.getvalue().splitlines()) == 2
