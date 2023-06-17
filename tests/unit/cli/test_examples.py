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
    workdir = next(tmp_path.glob("*"))

    # Check that copies of training script and metadata files have been generated
    assert (workdir / "simple_task.py").exists()
    assert (workdir / "simple_task.yaml").exists()
    xp_yaml = (workdir / "simple_task.yaml").read_text()
    assert "fairseq2.user.hubconf.CustomClass" in xp_yaml

    # Evaluation
    # Asserts the two ways of launching eval produce same results.
    snapshot = workdir / "epoch_2_step_100.train"
    assert snapshot.exists()
    eval_metrics = fairseq2.cli.commands.evaluate(
        workdir / "simple_task.py", snapshot="epoch_2_step_100.train", num_gpus=0
    )
    assert "eval/loss" in eval_metrics
    assert eval_metrics["eval/src_num_tokens"] == 20.0

    eval_metrics2 = fairseq2.cli.commands.evaluate(
        workdir / "epoch_2_step_100.train/hubconf.py", num_gpus=0
    )
    for k in ["loss", "src_num_tokens", "tgt_num_tokens"]:
        assert eval_metrics2[f"eval/{k}"] == eval_metrics[f"eval/{k}"]

    # Inference
    with monkeypatch.context() as m:
        m.setattr(sys, "stdin", io.StringIO("40 2\n25 25\n"))
        m.setattr(sys.stdin, "fileno", lambda: 128)
        stdout = io.StringIO()
        m.setattr(sys, "stdout", stdout)
        fairseq2.cli.commands.inference(snapshot, num_gpus=0)

    assert len(stdout.getvalue().splitlines()) == 2
