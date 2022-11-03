import dataclasses
import itertools
import typing as tp

from torchtnt.loggers.logger import MetricLogger, Scalar
from torchtnt.runner.callback import Callback
from torchtnt.runner.state import State
from torchtnt.runner.unit import TPredictUnit
from torchtnt.utils import get_global_rank


def _load_conf(config: tp.Any) -> tp.Any:
    """wandb.init doesn't handle dataclasses for config, convert them to
    dict."""
    if isinstance(config, dict):
        return {k: _load_conf(v) for k, v in config.items()}
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return config


class WandbLogger(MetricLogger):
    def __init__(self, project: str, config: tp.Any):
        self.project = project
        self.config = _load_conf(config)
        self.run_id: tp.Optional[str] = None
        self._rank: int = get_global_rank()
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
            resume="allow",
        )
        if self._wandb is not None:
            # wandb.init can fail
            self.run_id = self._wandb.id

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
        self._rank = get_global_rank()

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
    ) -> None:
        super().__init__()
        self.columns = header_row
        self.table_name = table_name
        self.logger = logger

    def get_batch_output_rows(
        self,
        state: State,
        unit: TPredictUnit[tp.Any],
        step_output: tp.Any,
    ) -> tp.List[tp.List[str]]:
        if isinstance(step_output, list):
            return step_output
        raise NotImplementedError()

    def on_predict_start(self, state: State, unit: TPredictUnit[tp.Any]) -> None:
        import wandb

        self.logger.prepare()
        self._table = wandb.Table(columns=self.columns, data=[])

    def on_predict_step_end(self, state: State, unit: TPredictUnit[tp.Any]) -> None:
        assert state.predict_state is not None
        step_output = state.predict_state.step_output
        batch_output_rows = self.get_batch_output_rows(state, unit, step_output)

        # Check whether the first item is a list or not
        for row in itertools.zip_longest(*batch_output_rows):
            self._table.add_row(*row)

    def on_predict_end(self, state: State, unit: TPredictUnit[tp.Any]) -> None:
        if state.train_state is None:
            step = 0
        else:
            step = state.train_state.progress.num_steps_completed
        self.logger._wandb.log(self.table_name, self._table, step=step)
