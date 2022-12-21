import dataclasses
import itertools
import logging
import typing as tp
from typing import Any, List

import torchtnt.utils.distributed
import torchtnt.utils.loggers
import torchtnt.utils.timer
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TEvalUnit, TTrainUnit
from torchtnt.utils.loggers import Scalar

from fairseq2.dataloader import Seq2SeqStr

log = logging.getLogger(__name__)


def _load_conf(config: tp.Any) -> tp.Any:
    """wandb.init doesn't handle dataclasses for config, convert them to
    dict."""
    if isinstance(config, dict):
        return {k: _load_conf(v) for k, v in config.items()}
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return config


class WandbLogger:
    def __init__(self, project: str, config: tp.Any):
        self.project = project
        self.config = _load_conf(config)
        self.run_id: tp.Optional[str] = None
        self._rank: int = torchtnt.utils.distributed.get_global_rank()
        self._wandb: tp.Any = None

    def prepare(self) -> None:
        if self._wandb is not None:
            return
        if self._rank != 0:
            return
        import wandb

        if "/" in self.project:
            entity, project = self.project.split("/", 1)
        else:
            entity, project = None, self.project

        self._wandb = wandb.init(
            project=project,
            entity=entity,
            id=self.run_id,
            config=self.config,
            # Note we don't want to force resuming.
            # Maybe you're resuming someone else run, and you don't have
            # access to their wandb experiment.
            resume="allow",
        )
        if self._wandb is None:
            # wandb.init can fail (it will already have printed a message)
            return

        self.run_id = self._wandb.id
        import __main__

        wandb.save(__main__.__file__, policy="now")

    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        return self.state_dict()

    def __setstate__(self, state: tp.Dict[str, tp.Any]) -> None:
        return self.load_state_dict(state)

    def state_dict(self) -> tp.Dict[str, tp.Any]:
        return {
            "project": self.project,
            "run_id": self.run_id,
            "config": self.config,
        }

    def load_state_dict(self, state_dict: tp.Dict[str, tp.Any]) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)
        self._wandb = None
        self._rank = torchtnt.utils.get_global_rank()

    def log(self, name: str, data: Scalar, step: int) -> None:
        raise NotImplementedError("Please use log_dict")

    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        """Log multiple scalar values.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int): step value to record
        """
        if self._rank != 0:
            return
        self.prepare()
        self._wandb.log(payload, step)

    def log_samples(self, samples: List[Seq2SeqStr], step: int, eval_step: int) -> None:
        import wandb

        # TODO: is it possible to upload partial tables from different workers ?
        # That would require coming up with a robust way of ID each sample.
        # Maybe on the dataloader itself ?
        if self._rank != 0 or eval_step > 0:
            return
        self.prepare()
        assert isinstance(samples, list)
        assert isinstance(samples[0], Seq2SeqStr)
        text_table = wandb.Table(
            columns=["id", "train_step", "source", "target", "predicted"]
        )
        for i, sample in enumerate(samples[:10]):
            text_table.add_data(i, step, *sample)
        self._wandb.log({"predictions": text_table})

    def close(self) -> None:
        """Close log resource, flushing if necessary.

        Logs should not be written after `close` is called.
        """
        if self._wandb is None:
            return
        self._wandb.finish()
        self._wandb = None


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
        table_name: str = "predictions",
        limit: int = 100,
    ) -> None:
        super().__init__()
        self.columns = header_row
        self.table_name = table_name
        self.logger = logger
        self.limit = limit
        self._world_size: int = torchtnt.utils.distributed.get_world_size()

    def get_batch_output_rows(
        self,
        state: State,
        unit: TEvalUnit[tp.Any],
        step_output: tp.Any,
    ) -> tp.List[tp.List[str]]:
        if isinstance(step_output, list):
            return step_output
        raise NotImplementedError()

    def on_eval_start(self, state: State, unit: TEvalUnit[tp.Any]) -> None:
        import wandb

        self.logger.prepare()
        self._table = wandb.Table(columns=self.columns, data=[])

    def on_eval_step_end(self, state: State, unit: TEvalUnit[tp.Any]) -> None:
        assert state.predict_state is not None
        step_output = state.predict_state.step_output
        batch_output_rows = self.get_batch_output_rows(state, unit, step_output)

        # Check whether the first item is a list or not
        for row in itertools.zip_longest(*batch_output_rows):
            self._table.add_row(*row)

    def on_eval_end(self, state: State, unit: TEvalUnit[tp.Any]) -> None:
        if state.train_state is None:
            step = 0
        else:
            step = state.train_state.progress.num_steps_completed
        self.logger._wandb.log(self.table_name, self._table, step=step)


class StdoutLogger:
    def log(self, name: str, data: Scalar, step: int) -> None:
        raise NotImplementedError("Please use log_dict")

    def log_dict(self, payload: tp.Mapping[str, Scalar], step: int) -> None:
        print("Step:", step, payload)

    def close(self) -> None:
        pass


class LogMetrics(Callback):
    def __init__(
        self,
        unit: TTrainUnit[Any],
        logger: torchtnt.utils.loggers.MetricLogger,
        frequency_steps: int = 100,
        sync_frequency: int = 1000,
    ):
        self.logger = logger
        self.frequency_steps = frequency_steps
        self.sync_frequency = sync_frequency
        self.global_rank = torchtnt.utils.get_global_rank()

    def on_train_step_end(self, state: State, unit: TTrainUnit[Any]) -> None:
        assert state.train_state is not None
        step = state.train_state.progress.num_steps_completed

        if step % self.frequency_steps != 0:
            return
        self.log_metrics(state, step, "train/", sync=step % self.sync_frequency == 0)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit[Any]) -> None:
        assert state.train_state is not None
        step = state.train_state.progress.num_steps_completed

        self.log_metrics(state, step, "train/", sync=True)

        self.logger.close()

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
        print(step, actual_metrics)

        report, total_calls, total_time = torchtnt.utils.timer._make_report(state.timer)
        for row in report[:10]:
            name, avg_duration, num_calls, total_duration, percentage = row
            actual_metrics[f"timer/{name}"] = percentage

        if self.global_rank == 0:
            self.logger.log_dict(actual_metrics, step)
        print(step, actual_metrics)
        metrics.reset()
