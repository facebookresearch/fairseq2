import dataclasses
import enum
import functools
import logging
import math
import os
import time
import typing as tp
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import torch
import torchtnt.utils.distributed
import torchtnt.utils.loggers
import torchtnt.utils.timer
import yaml
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TTrainUnit
from torchtnt.utils.loggers import Scalar
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger as TntTensorBoardLogger

log = logging.getLogger(__name__)

# TODO: This should be a helper somewhere
class Stateful:
    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        return self.state_dict()

    def __setstate__(self, state: tp.Dict[str, tp.Any]) -> None:
        return self.load_state_dict(state)

    def state_dict(self) -> tp.Dict[str, tp.Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def load_state_dict(self, state_dict: tp.Dict[str, tp.Any]) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)


class MetricLogger(Stateful):
    def __init__(self, config_file: Path):
        assert config_file.parent.is_dir()
        self.config_file = config_file
        self._rank: int = torchtnt.utils.distributed.get_global_rank()
        self._last_metrics: tp.Mapping[str, Scalar] = {}

    def on_train_start(self) -> None:
        job_info = collect_job_info()
        log.info(f"Job info: {job_info}")

        config = yaml.load(self.config_file.read_text(), Loader=yaml.Loader)
        if config is None:
            log.warning(f"{self.config_file} is empty")
            return

        config["job"] = job_info
        self.config_file.write_text(yaml.dump(config))
        print(config)

    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        ...

    def close(self) -> None:
        pass


def read_config(config_file: Path) -> Any:
    if not config_file.exists():
        return {}
    return yaml.load(config_file.read_text(), Loader=yaml.Loader)


def collect_job_info() -> Dict[str, str]:
    job = {}
    if "SLURM_JOB_ID" in os.environ:
        job["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    if "HOST" in os.environ:
        job["host"] = os.environ["HOST"]
    return job


class StdoutLogger(MetricLogger):
    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        self._last_metrics = payload
        if self._rank != 0:
            return
        print("Step:", step, _downgrade_for_stdout(payload))


def _downgrade_for_stdout(payload: tp.Mapping[str, tp.Any]) -> tp.Mapping[str, tp.Any]:
    """Converts a metric dict into something that can be logged by tensorboard"""

    def _downgrade(val: tp.Any) -> tp.Any:
        if isinstance(val, float):
            return round_sig(val, 4)
        if not isinstance(val, torch.Tensor):
            return val
        if val.ndim == 0:
            return round_sig(val.item(), 4)
        else:
            return f"tensor{tuple(val.shape)} on {val.device}"

    return {k: _downgrade(v) for k, v in payload.items()}


class WandbLogger(StdoutLogger):
    def __init__(
        self,
        config_file: Path,
        project: str,
        job_type: str,
        group_id: str,
    ):
        import wandb

        super().__init__(config_file)
        self.project = project
        self.group_id = group_id
        self.job_type = job_type
        self.wandb_id = "/".join((project, group_id))
        self._wandb = wandb

    @functools.cached_property
    def _wandb_run(self) -> "tp.Any":
        if self._rank != 0:
            return None

        if "/" in self.project:
            entity, project = self.project.split("/", 1)
        else:
            entity, project = None, self.project  # type: ignore

        config = read_config(self.config_file)
        run = self._wandb.init(
            project=project,
            entity=entity,
            id=f"{self.group_id}-{self.job_type}",
            group=self.group_id,
            job_type=self.job_type,
            resume="allow",
            config=_simple_conf(config),
        )

        if run is None:
            # wandb.init can fail (it will already have printed a message)
            return

        # This will specify the "entity"
        self.wandb_id = "/".join((run.entity, run.project, run.group))

        # We want to only do this once
        if "job" not in config:
            job = collect_job_info()
            job["wandb_id"] = self.wandb_id
            config["job"] = job
            # Update the config file with new information
            self.config_file.write_text(yaml.dump(config))
            print(config)
            self.upload_script_and_config()

        if self.job_type == "evaluate":
            # Tell W&B 'evalute' job is using the model from 'train' job.
            try:
                run.use_artifact(f"{self.group_id}:latest")
            except Exception:
                # The artifact may not be "ready" yet (not sure what that mean)
                pass

        return run

    def load_state_dict(self, state_dict: tp.Dict[str, tp.Any]) -> None:
        super().load_state_dict(state_dict)

        import wandb

        self._wandb = wandb

    def upload_script_and_config(self, top_secret: bool = False) -> None:
        """Uploads the script and config to W&B.

        When using `top_secret=True` only the file checksum will be uploaded.
        """
        if self._wandb_run is None:
            return

        artifact = self._wandb.Artifact(
            name=self.group_id,
            type="model",
            metadata=_simple_conf(read_config(self.config_file)),
        )
        script = self.config_file.with_suffix(".py")
        if top_secret:
            artifact.add_reference(f"file://{self.config_file}")
            artifact.add_reference(f"file://{script}")
        else:
            artifact.add_file(str(self.config_file))
            artifact.add_file(str(script))
            base_path = str(self.config_file.parent)
            self._wandb.save(str(self.config_file), policy="now", base_path=base_path)
            self._wandb.save(str(script), policy="now", base_path=base_path)
        self._wandb_run.log_artifact(artifact)

    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        """Log multiple scalar values.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int): step value to record
        """
        self._last_metrics = payload
        run = self._wandb_run
        if run is None:
            return
        run.log(_downgrade_for_tb(payload), step)

        # Also log to stdout
        super().log_dict(payload, step)

    def close(self) -> None:
        """Close log resource, flushing if necessary.

        Logs should not be written after `close` is called.
        """
        run = self._wandb_run
        if run is None:
            return
        run.finish()


class TensorBoardLogger(TntTensorBoardLogger, StdoutLogger):
    def __init__(self, config_file: Path, log_dir: tp.Optional[Path] = None):
        log_dir = log_dir or config_file.parent / "tb"
        log_dir.mkdir(exist_ok=True)
        super().__init__(str(log_dir))
        self.config_file = config_file

        if self._writer:
            config = _simple_conf(read_config(self.config_file))
            flat_conf = _flatten_dict(config, {})
            self._writer.add_hparams(flat_conf, {})

    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        TntTensorBoardLogger.log_dict(self, _downgrade_for_tb(payload), step)
        StdoutLogger.log_dict(self, payload, step)


def _downgrade_for_tb(payload: tp.Mapping[str, Scalar]) -> tp.Mapping[str, Scalar]:
    """Converts a metric dict into something that can be logged by tensorboard"""
    return {k: v for k, v in payload.items() if getattr(v, "ndim", 0) == 0}


class LogMetrics(Callback):
    def __init__(
        self,
        logger: MetricLogger,
        frequency_steps: int = 1000,
        sync_frequency: int = 1000,
        increased_freq_steps: int = 10,
        max_interval: timedelta = timedelta(minutes=20),
    ):
        """Logs the metrics every frequency_steps

        - sync_frequency: when to synchronize the metric across workers
        - increased_freq_steps: increase the frequency of logging at the beginning of training
        - max_interval: upper bound delay between two metric log
        """
        self.logger = logger
        self.frequency_steps = frequency_steps
        self.sync_frequency = max(sync_frequency, frequency_steps)
        self.increased_freq_steps = increased_freq_steps

        self.max_interval_seconds = max_interval.total_seconds()
        self._last_log = time.monotonic()

    def on_train_start(self, state: State, unit: TTrainUnit[Any]) -> None:
        if hasattr(self.logger, "on_train_start"):
            self.logger.on_train_start()

    def on_train_step_end(self, state: State, unit: TTrainUnit[Any]) -> None:
        assert state.train_state is not None
        step = state.train_state.progress.num_steps_completed
        freq = self.frequency_steps
        should_log = (step % freq == 0) or (
            # Increase the log frequency at the beginning of training
            step < self.increased_freq_steps * freq
            and step % (freq // self.increased_freq_steps) == 0
        )
        should_sync = step % self.sync_frequency == 0
        if (
            not should_log
            and time.monotonic() - self._last_log > self.max_interval_seconds
        ):
            # Long time, no log:
            # Force logging but without syncing because this may not trigger on all rank
            # at the same time.
            should_log, should_sync = True, False

        if should_log:
            self.log_metrics(state, step, "train/", sync=should_sync)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit[Any]) -> None:
        assert state.train_state is not None
        step = state.train_state.progress.num_steps_completed

        self.log_metrics(state, step, "train/", sync=True)

    def on_eval_end(self, state: State, unit: TEvalUnit[Any]) -> None:
        step = (
            0
            if state.train_state is None
            else state.train_state.progress.num_steps_completed
        )

        self.log_metrics(state, step, "eval/", sync=True)

    def log_metrics(self, state: State, step: int, prefix: str, sync: bool) -> None:
        # TODO: upgrade once torchtnt has builtin support for metric
        metrics = (
            state.eval_state.metrics if prefix == "eval/" else state.train_state.metrics  # type: ignore
        )
        actual_metrics = metrics.compute(sync=sync, prefix=prefix)

        report, total_calls, total_time = torchtnt.utils.timer._make_report(state.timer)
        for row in report[:10]:
            name, avg_duration, num_calls, total_duration, percentage = row
            if percentage < 1:
                continue
            actual_metrics[f"timer/{name}"] = percentage

        # Note: we call log_dict on all rank, we let the logger handle that.
        self.logger.log_dict(actual_metrics, step)

        metrics.reset()
        self._last_log = time.monotonic()


class WandbCsvWriter(Callback):
    """A callback to write prediction outputs to a W&B table. This reuse the
    torchtnt.BaseCSVWriter API.

    This callback provides an interface to simplify writing outputs during prediction
    into a CSV file. This callback must be extended with an implementation for
    ``get_batch_output_rows`` to write the desired outputs as rows in the CSV file.

    By default, outputs at each step across all processes will be written into the same CSV file.
    The outputs in each row is a a list of strings, and should match
    the columns names defined in ``header_row``.

    Args:
        header_row: name of the columns
        table_name: name of the table
    """

    def __init__(
        self,
        header_row: tp.List[str],
        logger: WandbLogger,
        table_name: str,
        limit: int = 1000,
    ) -> None:
        super().__init__()
        self.columns = header_row
        self.table_name = table_name
        self.logger = logger
        self.limit = limit
        self._table_size = 0
        self._world_size: int = torchtnt.utils.distributed.get_world_size()

    def get_batch_output_rows(
        self, step_output: tp.Any
    ) -> tp.Sequence[tp.Tuple[Any, ...]]:
        if isinstance(step_output, list):
            return step_output
        raise NotImplementedError()

    def _reset_table(self) -> None:
        import wandb

        _ = self.logger._wandb_run
        self._table = wandb.Table(columns=self.columns, data=[])
        self._table_size = 0

    def on_eval_start(self, state: State, unit: TEvalUnit[tp.Any]) -> None:
        self._reset_table()

    def on_train_start(self, state: State, unit: TTrainUnit[tp.Any]) -> None:
        self._reset_table()

    def on_train_step_end(self, state: State, unit: TTrainUnit[tp.Any]) -> None:
        if self._table_size > self.limit:
            return
        assert state.train_state is not None
        step_output = state.train_state.step_output
        batch_output_rows = self.get_batch_output_rows(step_output)

        # Check whether the first item is a list or not
        for row in batch_output_rows:
            self._table.add_data(*row)
            self._table_size += 1

        if self._table_size > self.limit:
            self.logger._wandb_run.log({self.table_name: self._table}, step=0)

    def on_eval_step_end(self, state: State, unit: TEvalUnit[tp.Any]) -> None:
        if self._table_size > self.limit:
            return

        assert state.eval_state is not None
        step_output = state.eval_state.step_output
        batch_output_rows = self.get_batch_output_rows(step_output)

        # Check whether the first item is a list or not
        for row in batch_output_rows:
            self._table.add_data(*row)
            self._table_size += 1

    def on_eval_end(self, state: State, unit: TEvalUnit[tp.Any]) -> None:
        if state.train_state is None:
            step = 0
        else:
            step = state.train_state.progress.num_steps_completed
        self.logger._wandb_run.log({self.table_name: self._table}, step=step)


def _simple_conf(config: tp.Any) -> tp.Any:
    """W&B doesn't handle dataclasses or NamedTuple for config, convert them to dict."""
    if hasattr(config, "_asdict"):
        config = config._asdict()
    if dataclasses.is_dataclass(config):
        config = dataclasses.asdict(config)
    if isinstance(config, dict):
        return {k: _simple_conf(v) for k, v in config.items()}
    if isinstance(config, enum.Enum):
        config = config.name
    if isinstance(config, Path):
        config = str(config)
    if isinstance(config, torch.device):
        config = str(config)
    if isinstance(config, torch.dtype):
        config = str(config)
    return config


def _flatten_dict(
    config: Dict[str, tp.Any],
    res: Dict[str, tp.Union[int, float, str]],
    prefix: str = "",
) -> Dict[str, tp.Union[int, float, str]]:
    for k, v in config.items():
        full_key = "/".join((prefix, k)) if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, res, prefix=full_key)
        elif isinstance(v, (int, float, str)):
            res[full_key] = v
    return res


_SPECIAL_FLOATS = (0.0, float("inf"), float("-inf"))


def round_sig(x: float, sig: int) -> float:
    if math.isnan(x) or x in _SPECIAL_FLOATS:
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))
