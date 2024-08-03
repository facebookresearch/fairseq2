# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from itertools import count
from pathlib import Path
from statistics import mean
from typing import Any, Generic, TypeVar, final

import torch
import torch.distributed
from rich.progress import Progress, TaskID
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.profiler import record_function
from torcheval.metrics import Mean
from typing_extensions import override

from fairseq2.checkpoint import CheckpointManager, CheckpointNotFoundError
from fairseq2.datasets import DataReader
from fairseq2.early_stopper import EarlyStopper
from fairseq2.gang import FakeGang, Gang, all_sum, broadcast_flag
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    JsonFileMetricRecorder,
    LogMetricRecorder,
    MetricBag,
    MetricRecorder,
    TensorBoardRecorder,
    format_metric_value,
    record_metrics,
)
from fairseq2.nn.fsdp import summon_fsdp_for_validation
from fairseq2.nn.utils.gradient import (
    check_gradient_norms,
    clip_gradient_norm,
    normalize_gradients,
)
from fairseq2.optim import DynamicLossScaler
from fairseq2.optim.lr_scheduler import LRScheduler, NoopLR, get_effective_lr
from fairseq2.recipes.common_metrics import set_throughput_value
from fairseq2.recipes.evaluator import EvalUnit
from fairseq2.recipes.utils.cli import create_rich_progress
from fairseq2.typing import CPU, DataType
from fairseq2.utils.profiler import Profiler, Stopwatch
from fairseq2.utils.rng import RngBag
from fairseq2.utils.state import FSDPOptimizerStateHandler, StatefulObjectBag

log = get_log_writer(__name__)


BatchT = TypeVar("BatchT")

BatchT_contra = TypeVar("BatchT_contra", contravariant=True)


class TrainUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Trainer`."""

    @abstractmethod
    def __call__(self, batch: BatchT_contra) -> tuple[Tensor, int]:
        """Process ``batch``.

        :returns:
            The loss and the number of targets used to compute the loss.
        """

    @abstractmethod
    def set_step_nr(self, step_nr: int) -> None:
        """Set the current training step number."""

    @property
    @abstractmethod
    def model(self) -> Module:
        """The underlying model."""

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag:
        """The training-related metrics."""


class AbstractTrainUnit(TrainUnit[BatchT]):
    """Provides a skeletal implementation of :class:`TrainUnit`."""

    def __init__(self, model: Module) -> None:
        self._model = model

    @override
    def set_step_nr(self, step_nr: int) -> None:
        pass

    @final
    @property
    @override
    def model(self) -> Module:
        return self._model


@final
class Trainer(StatefulObjectBag, Generic[BatchT]):
    """Trains a machine learning model."""

    _model: Module
    _unit: TrainUnit[BatchT]
    _data_reader: DataReader[BatchT]
    _root_gang: Gang
    _dp_gang: Gang
    _tp_gang: Gang
    _dtype: DataType
    _optimizer: Optimizer
    _lr_scheduler: LRScheduler
    _loss_scaler: DynamicLossScaler
    _max_gradient_norm: float | None
    _step_nr: int
    _max_num_steps: int | None
    _data_epoch_nr: int
    _max_num_data_epochs: int | None
    _repeat_step: bool
    _read_data: bool
    _end_of_data_epoch: bool
    _end_of_data: bool
    _should_stop: bool
    _score_metric_name: str | None
    _lower_better: bool
    _early_stopper: EarlyStopper | None
    _best_step_and_score: tuple[int, float] | None
    _valid_score: float | None
    _valid_units: Sequence[EvalUnit[BatchT]]
    _valid_data_readers: Sequence[DataReader[BatchT]]
    _validate_after_n_steps: int
    _validate_every_n_steps: int | None
    _validate_after_n_data_epochs: int
    _validate_every_n_data_epochs: int | None
    _checkpoint_manager: CheckpointManager
    _checkpoint_after_n_steps: int
    _checkpoint_every_n_steps: int | None
    _checkpoint_after_n_data_epochs: int
    _checkpoint_every_n_data_epochs: int | None
    _keep_last_n_checkpoints: int | None
    _keep_best_n_checkpoints: int | None
    _keep_last_n_models: int | None
    _keep_best_n_models: int | None
    _metric_bag: MetricBag
    _metric_recorders: list[MetricRecorder]
    _publish_metrics_after_n_steps: int
    _publish_metrics_every_n_steps: int | None
    _publish_metrics_after_n_data_epochs: int
    _publish_metrics_every_n_data_epochs: int | None
    _profiler: Profiler
    _anomaly_detection: bool
    _seed: int
    _rng_bag: RngBag
    _wall_watch: Stopwatch
    _step_time: float
    _run: bool
    _progress: Progress
    _train_task_id: TaskID

    def __init__(
        self,
        *,
        unit: TrainUnit[BatchT],
        data_reader: DataReader[BatchT],
        root_gang: Gang,
        optimizer: Optimizer,
        checkpoint_manager: CheckpointManager,
        wall_watch: Stopwatch,
        dtype: DataType = torch.float32,
        dp_gang: Gang | None = None,
        tp_gang: Gang | None = None,
        lr_scheduler: LRScheduler | None = None,
        fp16_loss_scale: tuple[float, float] = (128.0, 0.0001),
        max_gradient_norm: float | None = None,
        max_num_steps: int | None = None,
        max_num_data_epochs: int | None = None,
        score_metric_name: str | None = None,
        lower_better: bool = False,
        early_stopper: EarlyStopper | None = None,
        valid_units: Sequence[EvalUnit[BatchT]] | None = None,
        valid_data_readers: Sequence[DataReader[BatchT]] | None = None,
        validate_after_n_steps: int = 0,
        validate_every_n_steps: int | None = None,
        validate_after_n_data_epochs: int = 0,
        validate_every_n_data_epochs: int | None = None,
        checkpoint_after_n_steps: int = 0,
        checkpoint_every_n_steps: int | None = None,
        checkpoint_after_n_data_epochs: int = 0,
        checkpoint_every_n_data_epochs: int | None = None,
        keep_last_n_checkpoints: int | None = None,
        keep_best_n_checkpoints: int | None = None,
        keep_last_n_models: int | None = None,
        keep_best_n_models: int | None = None,
        tb_dir: Path | None = None,
        metrics_dir: Path | None = None,
        publish_metrics_after_n_steps: int = 0,
        publish_metrics_every_n_steps: int | None = None,
        publish_metrics_after_n_data_epochs: int = 0,
        publish_metrics_every_n_data_epochs: int | None = None,
        profile: tuple[int, int] | None = None,
        anomaly_detection: bool = False,
        seed: int = 2,
    ) -> None:
        """
        :param unit:
            The training unit.
        :param data_reader:
            The data reader for training.
        :param root_gang:
            The gang for distributed training.
        :param optimizer:
            The parameter optimizer.
        :param checkpoint_manager:
            The checkpoint manager.
        :param wall_watch:
            The stopwatch to track process wall-time.
        :param dtype:
            The data type to train with.
        :param dp_gang:
            The data parallel gang. If ``None``, ``gang`` will be used.
        :param tp_gang:
            The tensor parallel gang. Only required for tensor parallel models.
        :param lr_scheduler:
            The learning rate scheduler.
        :param fp16_loss_scale:
            The initial and minimum loss scale for fp16 training.
        :param max_gradient_norm:
            The maximum gradient norm. If ``None``, no clipping will be applied.
        :param max_num_steps:
            The maximum number of steps to train for.
        :param max_num_data_epochs:
            The maximum number of data epochs to train for.
        :param score_metric_name:
            The name of the metric to use for score calculation.
        :param lower_better:
            If ``True``, lower scores are considered better.
        :param early_stopper:
            The early-stopper callable.
        :param valid_units:
            The evaluation units for validating the model.
        :param valid_data_readers:
            The data readers corresponding to each unit in ``valid_units``.
        :param validate_after_n_steps:
            The number of steps after which to start validating the model.
        :param validate_every_n_steps:
            The step interval at which to validate the model.
        :param validate_after_n_data_epochs:
            The number of data epochs after which to start validating the model.
        :param validate_every_n_data_epochs:
            The data epoch interval at which to validate the model.
        :param checkpoint_after_n_steps:
            The number of steps after which to start checkpointing.
        :param checkpoint_every_n_steps:
            The step interval at which to checkpoint.
        :param checkpoint_after_n_data_epochs:
            The number of data epochs after which to start checkpointing.
        :param checkpoint_every_n_data_epochs:
            The data epoch interval at which to checkpoint.
        :param keep_last_n_checkpoints:
            The number of checkpoints to keep. If ``None``, none will be deleted.
        :param keep_best_n_checkpoints:
            The number of checkpoints to keep based on their validation score.
            If ``None``, none will be deleted.
        :param keep_last_n_models:
            The number of checkpoint models to keep. Must be greater than or
            equal to ``keep_last_n_checkpoints``.
        :param keep_best_n_models:
            The number of best checkpoint models to keep based on their
            validation score. Must be greater than or equal to
            ``keep_best_n_checkpoints``.
        :param tb_dir:
            The TensorBoard log directory to dump metrics.
        :param metrics_dir:
            The directory to dump metrics.
        :param publish_metrics_after_n_steps:
            The number of steps after which to start publishing metrics.
        :param publish_metrics_every_n_steps:
            The step interval at which to publish metrics.
        :param publish_metrics_after_n_data_epochs:
            The number of data epochs after which to start publishing metrics.
        :param publish_metrics_every_n_data_epochs:
            The data epoch interval at which to publish metrics.
        :param profile:
            The number of steps that the PyTorch profiler should skip and then
            record.
        :param anomaly_detection:
            If ``True``, turns on anomaly detection feature in ``torch.autograd``.
        :param seed:
            The random number generator seed.
        """
        super().__init__()

        device = root_gang.device

        self._model = unit.model

        self._unit = unit

        self._data_reader = data_reader

        self._root_gang = root_gang

        if dp_gang is not None and tp_gang is not None:
            self._dp_gang = dp_gang
            self._tp_gang = tp_gang
        elif dp_gang is None and tp_gang is None:
            self._dp_gang = root_gang
            self._tp_gang = FakeGang(device=device)
        else:
            raise ValueError("`dp_gang` and `tp_gang` must be both specified.")

        if root_gang.rank == 0:
            if self._dp_gang.rank != 0 or self._tp_gang.rank != 0:
                raise ValueError(
                    f"The coordinator process of `root_gang` (i.e. rank 0) must be rank 0 in `dp_gang` and `tp_gang`, but is {self._dp_gang.rank} and {self._tp_gang.rank} instead."
                )

        self._dtype = dtype

        if uses_fsdp := isinstance(self._model, FSDP):
            self.register_stateful(
                "_optimizer", optimizer, FSDPOptimizerStateHandler(self._model)
            )
        else:
            self._optimizer = optimizer

        self._lr_scheduler = lr_scheduler or NoopLR(optimizer)

        fp16_init_scale, fp16_min_scale = fp16_loss_scale

        self._loss_scaler = DynamicLossScaler(
            optimizer,
            root_gang,
            sharded=uses_fsdp or self._tp_gang.size > 0,
            init_scale=fp16_init_scale,
            min_scale=fp16_min_scale,
            gradient_accumulation=self._data_reader.num_accumulate,
            enabled=self._dtype == torch.float16,
        )

        self._max_gradient_norm = max_gradient_norm

        self.register_stateful("_step_nr", 0)

        if max_num_steps == 0:
            raise ValueError("`max_num_steps` must be greater than zero.")

        self._max_num_steps = max_num_steps

        self.register_stateful("_data_epoch_nr", 1)

        if max_num_data_epochs == 0:
            raise ValueError("`max_num_data_epochs` must be greater than zero.")

        self._max_num_data_epochs = max_num_data_epochs

        self._repeat_step = False

        self._read_data = False  # Indicates whether we have read any data.

        self._end_of_data_epoch = False
        self._end_of_data = False

        self._should_stop = False

        self._score_metric_name = score_metric_name

        self._lower_better = lower_better

        if early_stopper is not None:
            if score_metric_name is None:
                raise ValueError(
                    "`score_metric_name` must be specified when `early_stopper` is specified."
                )

            if root_gang.rank != 0:
                early_stopper = lambda step_nr, score: False

            self._early_stopper = early_stopper
        else:
            self._early_stopper = None

        self.register_stateful("_best_step_and_score", None)

        self._valid_score = None

        if valid_units is None and valid_data_readers is None:
            self._valid_units = []

            self._valid_data_readers = []
        elif valid_units is not None and valid_data_readers is not None:
            if len(valid_units) != len(valid_data_readers):
                raise ValueError(
                    f"The number of data readers in `valid_data_readers` must match the number of units in `valid_units` ({len(valid_units)}), but is {len(valid_data_readers)} instead."
                )

            self._valid_units = valid_units

            self._valid_data_readers = valid_data_readers
        else:
            raise ValueError(
                "`valid_units` and `valid_data_readers` must be both specified."
            )

        if validate_every_n_steps == 0:
            raise ValueError("`validate_every_n_steps` must be greater than zero.")

        self._validate_after_n_steps = validate_after_n_steps
        self._validate_every_n_steps = validate_every_n_steps

        if validate_every_n_data_epochs == 0:
            raise ValueError(
                "`validate_every_n_data_epochs` must be greater than zero."
            )

        self._validate_after_n_data_epochs = validate_after_n_data_epochs
        self._validate_every_n_data_epochs = validate_every_n_data_epochs

        self._checkpoint_manager = checkpoint_manager

        if checkpoint_every_n_steps == 0:
            raise ValueError("`checkpoint_every_n_steps` must be greater than zero.")

        self._checkpoint_after_n_steps = checkpoint_after_n_steps
        self._checkpoint_every_n_steps = checkpoint_every_n_steps

        if checkpoint_every_n_data_epochs == 0:
            raise ValueError(
                "`checkpoint_every_n_data_epochs` must be greater than zero."
            )

        self._checkpoint_after_n_data_epochs = checkpoint_after_n_data_epochs
        self._checkpoint_every_n_data_epochs = checkpoint_every_n_data_epochs

        if keep_last_n_checkpoints is not None and keep_best_n_checkpoints is not None:
            raise ValueError(
                "`keep_last_n_checkpoints` and `keep_best_n_checkpoints` are mutually exclusive and must not be specified at the same time."
            )

        if keep_last_n_checkpoints == 0:
            raise ValueError("`keep_last_n_checkpoints` must be greater than zero.")

        if keep_best_n_checkpoints == 0:
            raise ValueError("`keep_best_n_checkpoints` must be greater than zero.")

        if keep_best_n_checkpoints is not None:
            if checkpoint_every_n_steps is not None:
                if score_metric_name is None:
                    raise ValueError(
                        "`score_metric_name` must be specified when `keep_best_n_checkpoints` is specified."
                    )

                if validate_every_n_steps is None:
                    raise ValueError(
                        "`validate_every_n_steps` must be specified when `keep_best_n_checkpoints` is specified."
                    )

                if checkpoint_every_n_steps % validate_every_n_steps != 0:
                    raise ValueError(
                        f"`checkpoint_every_n_steps` must be a multiple of `validate_every_n_steps` ({validate_every_n_steps}) when `keep_best_n_checkpoints` is specified, but is {checkpoint_every_n_steps} instead."
                    )

        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._keep_best_n_checkpoints = keep_best_n_checkpoints

        if keep_last_n_models is not None:
            if keep_last_n_checkpoints is None:
                raise ValueError(
                    "`keep_last_n_models` must be `None` when `keep_last_n_checkpoints` is `None`."
                )

            if keep_last_n_checkpoints > keep_last_n_models:
                raise ValueError(
                    f"`keep_last_n_models` must be greater than or equal to `keep_last_n_checkpoints` ({keep_last_n_checkpoints}), but is {keep_last_n_models} instead."
                )

        if keep_best_n_models is not None:
            if keep_best_n_checkpoints is None:
                raise ValueError(
                    "`keep_best_n_models` must be `None` when `keep_best_n_checkpoints` is `None`."
                )

            if keep_best_n_checkpoints > keep_best_n_models:
                raise ValueError(
                    f"`keep_best_n_models` must be greater than or equal to `keep_best_n_checkpoints` ({keep_best_n_checkpoints}), but is {keep_best_n_models} instead."
                )

        self._keep_last_n_models = keep_last_n_models
        self._keep_best_n_models = keep_best_n_models

        unit.metric_bag.register_metric(
            "gradient_norm", Mean(device=device), persistent=False
        )

        self._metric_bag = unit.metric_bag

        if root_gang.rank == 0:
            self._metric_recorders = [LogMetricRecorder(log)]

            if tb_dir is not None:
                self._metric_recorders.append(TensorBoardRecorder(tb_dir))

            if metrics_dir is not None:
                self._metric_recorders.append(JsonFileMetricRecorder(metrics_dir))
        else:
            self._metric_recorders = []

        if publish_metrics_every_n_steps == 0:
            raise ValueError(
                "`publish_metrics_every_n_steps` must be greater than zero."
            )

        self._publish_metrics_after_n_steps = publish_metrics_after_n_steps
        self._publish_metrics_every_n_steps = publish_metrics_every_n_steps

        if publish_metrics_every_n_data_epochs == 0:
            raise ValueError(
                "`publish_metrics_every_n_data_epochs` must be greater than zero."
            )

        self._publish_metrics_after_n_data_epochs = publish_metrics_after_n_data_epochs
        self._publish_metrics_every_n_data_epochs = publish_metrics_every_n_data_epochs

        if profile is None or tb_dir is None:
            if profile is not None and tb_dir is None:
                log.warning("No TensorBoard log directory provided. Profiling will be disabled.")  # fmt: skip

            skip_first, active_steps = 1, 0

            profile_dir = Path()
        else:
            skip_first, active_steps = profile

            profile_dir = tb_dir

        self._profiler = Profiler(
            skip_first, active_steps, profile_dir, root_gang, enabled=active_steps > 0
        )

        self._anomaly_detection = anomaly_detection

        self._seed = seed

        self._rng_bag = RngBag.from_device_defaults(CPU, device)

        self._wall_watch = wall_watch

        self._step_time = 0.0

        self._run = False

        self._progress = create_rich_progress()

    def request_stop(self) -> None:
        """Request a graceful stop of the training."""
        log.info("Stopping training after a final validation and saving checkpoint.")

        self._should_stop = True

    def __call__(self) -> None:
        if self._run:
            raise RuntimeError("The trainer can only be run once.")

        self._run = True

        # Set the per-rank seed for training.
        self._rng_bag.manual_seed(self._seed + self._root_gang.rank)

        try:
            self._maybe_restore_state()
        except KeyboardInterrupt:
            log.info("Training terminated!")

            raise

        log.info("Running training on {} device(s).", self._root_gang.size)

        try:
            self._do_run()
        except KeyboardInterrupt:
            log.info("Training terminated at step {}!", self._step_nr)

            raise

        if self._should_stop:
            log.info("Training stopped at step {}!", self._step_nr)

            return

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Training complete in {:,} seconds after {} step(s)!", int(elapsed_time), self._step_nr)  # fmt: skip

    def _maybe_restore_state(self) -> None:
        log.info("Attempting to load the last checkpoint.")

        try:
            step_nr, state = self._checkpoint_manager.load_last_checkpoint()
        except CheckpointNotFoundError:
            log.info("No checkpoint found. Starting training.")

            return

        log.info("Checkpoint loaded, restoring training from step {}.", step_nr)

        self._step_nr = step_nr

        self.load_state_dict(state)

        self._root_gang.barrier()

        log.info("Training restored, resuming.")

    def _do_run(self) -> None:
        with self._progress, self._profiler:
            self._train_task_id = self._progress.add_task(
                "train", total=self._max_num_steps, completed=self._step_nr
            )

            while self._should_run_step():
                self._maybe_advance_data_epoch()

                self._advance_step()

                detect_anomaly = torch.autograd.set_detect_anomaly(  # type: ignore[attr-defined]
                    self._anomaly_detection, check_nan=True
                )

                with detect_anomaly:
                    with record_function(f"step_{self._step_nr}"):
                        self._run_step()

                if self._should_publish_metrics():
                    self._publish_metrics()

                if self._should_validate():
                    self._validate()

                    self._maybe_request_early_stop()

                if self._should_checkpoint():
                    self._checkpoint()

                self._profiler.step()

                self._valid_score = None

    def _should_run_step(self) -> bool:
        if self._end_of_data or self._should_stop:
            return False

        if self._max_num_steps is None:
            return True

        return self._step_nr < self._max_num_steps

    def _maybe_advance_data_epoch(self) -> None:
        if not self._end_of_data_epoch:
            return

        self._data_epoch_nr += 1

        self._end_of_data_epoch = False

    def _advance_step(self, value: int = 1) -> None:
        self._step_nr += value

        self._progress.update(self._train_task_id, advance=value)

    def _run_step(self) -> None:
        step_nr = self._step_nr

        log.debug("{} training step {}.", "Repeating" if self._repeat_step else "Running", step_nr)  # fmt: skip

        watch = Stopwatch(start=True, device=self._root_gang.device)

        # Collect the batches.
        with record_function(f"step_{step_nr}_data_load"):
            try:
                batches = self._next_batches()
            except StopIteration:
                return

        # Prepare the unit.
        if not self._repeat_step:
            with record_function(f"step_{step_nr}_unit_setup"):
                self._unit.set_step_nr(step_nr)

        num_targets = 0

        if self._loss_scaler.is_enabled:
            self._metric_bag.begin_updates()

        # Accumulate.
        for batch_nr, batch in enumerate(batches):
            with self._maybe_no_sync(batch_nr, len(batches)):
                with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                    batch_loss, num_batch_targets = self._compute_loss(batch)

                with record_function(f"step_{step_nr}_{batch_nr}_backward"):
                    self._loss_scaler.backward(batch_loss)

            num_targets += num_batch_targets

        # Normalize.
        normalize_gradients(self._model, self._dp_gang, num_targets=num_targets)

        # Clip.
        with record_function(f"step_{step_nr}_grad_norm"):
            self._loss_scaler.unscale_gradients_()

            # TODO(balioglu): Support tensor parallelism!
            grad_norm = clip_gradient_norm(
                self._model, max_norm=self._max_gradient_norm
            )

            # Sanity check.
            if not check_gradient_norms(grad_norm, self._dp_gang, step_nr):
                raise FloatingPointError(
                    f"The gradients are inconsistent between processes at step {step_nr}. Training cannot continue."
                )

        # Update the parameters.
        with record_function(f"step_{step_nr}_optimizer"):
            _, scale_result = self._loss_scaler.run_optimizer_step(step_nr)

        if scale_result.overflow:
            self._metric_bag.rollback_updates()

            if scale_result.min_reached:
                raise FloatingPointError(
                    f"The gradients are scaled down to minimum at step {step_nr}. Training cannot continue."
                )

            self._advance_step(-1)

            self._repeat_step = True
        else:
            self._lr_scheduler.step()

            if self._loss_scaler.is_enabled:
                self._metric_bag.commit_updates()

            self._metric_bag.gradient_norm.update(grad_norm)

            self._repeat_step = False

        # Reset.
        self._optimizer.zero_grad(set_to_none=True)

        self._step_time += watch.get_elapsed_time()

    def _next_batches(self) -> list[BatchT]:
        try:
            batches = next(self._data_reader)

            self._read_data = True

            return batches
        except StopIteration:
            pass

        self._end_of_data_epoch = True

        log.info("End of epoch {} reached at training step {}.", self._data_epoch_nr, self._step_nr)  # fmt: skip

        if not self._read_data:  # Means the dataset is empty.
            self._end_of_data = True
        elif self._max_num_data_epochs is not None:
            if self._data_epoch_nr >= self._max_num_data_epochs:
                self._end_of_data = True

        if self._end_of_data:
            log.info("End of data reached.", self._step_nr)
        else:
            self._repeat_step = True

        self._advance_step(-1)

        self._data_reader.reset()

        raise StopIteration()

    def _maybe_no_sync(
        self, batch_nr: int, num_batches: int
    ) -> AbstractContextManager[None]:
        if batch_nr < num_batches - 1 and self._dp_gang.size > 1:
            return self._model.no_sync()  # type: ignore[no-any-return]

        return nullcontext()

    def _compute_loss(self, batch: BatchT) -> tuple[Tensor, int]:
        with self._maybe_autocast():
            return self._unit(batch)

    def _maybe_autocast(self) -> AbstractContextManager[None]:
        if self._dtype == torch.float32:
            return nullcontext()

        if self._model.training and isinstance(self._model, (DDP, FSDP)):
            if self._model.mixed_precision is not None:
                return nullcontext()

        return torch.autocast(device_type=self._dp_gang.device.type, dtype=self._dtype)

    def _should_publish_metrics(self) -> bool:
        after_n_steps = self._publish_metrics_after_n_steps
        every_n_steps = self._publish_metrics_every_n_steps

        if self._should_do_at_step(after_n_steps, every_n_steps):
            return True

        after_n_data_epochs = self._publish_metrics_after_n_data_epochs
        every_n_data_epochs = self._publish_metrics_every_n_data_epochs

        return self._should_do_at_data_epoch(after_n_data_epochs, every_n_data_epochs)

    def _publish_metrics(self) -> None:
        log.debug("Syncing metrics.")

        if self._tp_gang.rank == 0:
            values = self._metric_bag.sync_and_compute_metrics()
        else:
            values = None

        self._metric_bag.reset_non_persistent_metrics()

        elapsed_time = self._step_time

        self._step_time = 0.0

        if self._root_gang.rank != 0:
            return

        assert values is not None

        values["lr"] = get_effective_lr(self._lr_scheduler)

        set_throughput_value(values, elapsed_time)

        values["elapsed_time"] = elapsed_time

        values["wall_time"] = self._wall_watch.get_elapsed_time()

        record_metrics(self._metric_recorders, "train", values, self._step_nr)

    def _should_validate(self) -> bool:
        if not self._valid_units:
            return False

        after_n_steps = self._validate_after_n_steps
        every_n_steps = self._validate_every_n_steps

        if self._should_do_at_step(after_n_steps, every_n_steps):
            return True

        after_n_data_epochs = self._validate_after_n_data_epochs
        every_n_data_epochs = self._validate_every_n_data_epochs

        return self._should_do_at_data_epoch(after_n_data_epochs, every_n_data_epochs)

    @torch.inference_mode()
    def _validate(self) -> None:
        log.info("Starting validation after step {}.", self._step_nr)

        with summon_fsdp_for_validation(self._model):
            self._model.eval()

            unit_scores = []

            for unit, data_reader in zip(self._valid_units, self._valid_data_readers):
                if unit.display_name:
                    log.info("Validating {}.", unit.display_name)

                unit_score = self._validate_unit(unit, data_reader)
                if unit_score is not None:
                    unit_scores.append(unit_score)

            self._valid_score = self._compute_valid_score(unit_scores)

            self._model.train()

        log.info("Validation complete.")

    def _validate_unit(
        self, unit: EvalUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> float | None:
        watch = Stopwatch(start=True, device=self._root_gang.device)

        unit.model.eval()

        unit.set_step_nr(self._step_nr)

        valid_task = self._progress.add_task("valid", total=None)

        for step_nr in count(start=1):
            self._progress.update(valid_task, advance=1)

            log.debug("Running validation step {}.", step_nr)

            try:
                batches = next(data_reader)
            except StopIteration:
                batches = []

            for batch in batches:
                with self._maybe_autocast():
                    unit(batch)

            if self._is_valid_end_of_data(batches):
                break

        self._progress.remove_task(valid_task)

        data_reader.reset()

        time = watch.get_elapsed_time()

        metric_values = self._publish_validation_metrics(unit, time)

        return self._get_unit_score(metric_values)

    def _is_valid_end_of_data(self, batches: list[BatchT]) -> bool:
        total_num_batches = all_sum(self._dp_gang, len(batches))

        return bool(total_num_batches == 0)

    def _publish_validation_metrics(
        self, unit: EvalUnit[BatchT], elapsed_time: float
    ) -> dict[str, Any] | None:
        log.debug("Syncing validation metrics.")

        if self._tp_gang.rank == 0:
            values = unit.metric_bag.sync_and_compute_metrics()
        else:
            values = None

        unit.metric_bag.reset_metrics()

        if self._root_gang.rank != 0:
            return None

        assert values is not None

        set_throughput_value(values, elapsed_time)

        values["elapsed_time"] = elapsed_time

        values["wall_time"] = self._wall_watch.get_elapsed_time()

        if unit.display_name:
            run_name = "valid/" + unit.display_name
        else:
            run_name = "valid"

        record_metrics(self._metric_recorders, run_name, values, self._step_nr)

        return values

    def _get_unit_score(self, metric_values: dict[str, Any] | None) -> float | None:
        if metric_values is None:
            return None

        if self._score_metric_name is None:
            return None

        score = metric_values.get(self._score_metric_name)
        if score is None:
            return None

        if not isinstance(score, (int, float, Tensor)):
            log.warning("The score metric must be of type `int`, `float`, or `torch.Tensor`.")  # fmt: skip

            return None

        return float(score)

    def _compute_valid_score(self, unit_scores: list[float]) -> float | None:
        if self._score_metric_name is None:
            return None

        if not unit_scores:
            if self._root_gang.rank == 0:
                raise RuntimeError(
                    "None of the validation units returned a score metric value. Please file a bug report to the recipe author."
                )

            return None

        score = mean(unit_scores)

        def is_better_score() -> bool:
            if self._best_step_and_score is None:
                return True

            best_score = self._best_step_and_score[1]

            return best_score > score if self._lower_better else best_score < score

        if is_better_score():
            self._best_step_and_score = (self._step_nr, score)

        if log.is_enabled_for_info():
            best_step_nr, best_score = self._best_step_and_score  # type: ignore[misc]

            if len(unit_scores) > 1:
                m1 = "Mean "
                m2 = "Best Mean "
            else:
                m1 = ""
                m2 = "Best "

            s1 = format_metric_value(self._score_metric_name, score)
            s2 = format_metric_value(self._score_metric_name, best_score)

            log.info("Score (step {}) - {}{} | {}{} at step {}", self._step_nr, m1, s1, m2, s2, best_step_nr)  # fmt: skip

        return score

    def _maybe_request_early_stop(self) -> None:
        if self._early_stopper is None:
            return

        if self._root_gang.rank == 0:
            assert self._valid_score is not None

            should_stop = self._early_stopper(self._step_nr, self._valid_score)
        else:
            should_stop = False

        self._should_stop = broadcast_flag(self._root_gang, should_stop)

        if self._should_stop:
            log.info("Early stop requested. Training will be terminated after saving checkpoint.")  # fmt: skip

    def _should_checkpoint(self) -> bool:
        after_n_steps = self._checkpoint_after_n_steps
        every_n_steps = self._checkpoint_every_n_steps

        if self._should_do_at_step(after_n_steps, every_n_steps):
            return True

        after_n_data_epochs = self._checkpoint_after_n_data_epochs
        every_n_data_epochs = self._checkpoint_every_n_data_epochs

        return self._should_do_at_data_epoch(after_n_data_epochs, every_n_data_epochs)

    def _checkpoint(self) -> None:
        step_nr = self._step_nr

        log.info("Saving checkpoint after step {}.", step_nr)

        log.info("Extracting trainer state.")

        state = self.state_dict()

        log.info("Trainer state extracted.")

        self._checkpoint_manager.begin_checkpoint(step_nr)

        log.info("Saving trainer state.")

        if self._dp_gang.size > 1 and isinstance(self._model, DDP):
            replicated_keys = {"_model", "_optimizer"}
        else:
            replicated_keys = None

        self._checkpoint_manager.save_state(
            state, model_key="_model", replicated_keys=replicated_keys
        )

        log.info("Trainer state saved.")

        if self._score_metric_name is not None:
            log.info("Saving checkpoint score.")

            self._checkpoint_manager.save_score(self._valid_score)

            log.info("Checkpoint score saved.")

        if isinstance(self._model, FSDP):
            log.info("Saving consolidated FSDP model.")

            self._checkpoint_manager.save_consolidated_fsdp_model(self._model)

            log.info("Consolidated FSDP model saved.")

        self._checkpoint_manager.commit_checkpoint()

        log.info("Checkpoint complete.")

        # Clean up the checkpoints.
        nc = self._keep_last_n_checkpoints
        nm = self._keep_last_n_models

        if nm is not None:
            assert nc is not None

            self._checkpoint_manager.keep_last_n_checkpoints(nm)
            self._checkpoint_manager.keep_last_n_checkpoints(nc, preserve_model=True)
        elif nc is not None:
            self._checkpoint_manager.keep_last_n_checkpoints(nc)

        nc = self._keep_best_n_checkpoints
        nm = self._keep_best_n_models

        if nm is not None:
            assert nc is not None

            self._checkpoint_manager.keep_best_n_checkpoints(nm)
            self._checkpoint_manager.keep_best_n_checkpoints(nc, preserve_model=True)
        elif nc is not None:
            self._checkpoint_manager.keep_best_n_checkpoints(nc)

    def _should_do_at_step(self, after_n_steps: int, n_steps: int | None) -> bool:
        if self._end_of_data or self._should_stop:
            return self._read_data

        if self._repeat_step:
            return False

        if self._max_num_steps is not None:
            if self._step_nr >= self._max_num_steps:
                return True

        if n_steps is None:
            return False

        if self._step_nr < after_n_steps:
            return False

        return self._step_nr % n_steps == 0

    def _should_do_at_data_epoch(
        self, after_n_data_epochs: int, n_data_epochs: int | None
    ) -> bool:
        if self._end_of_data or self._should_stop:
            return self._read_data

        if not self._end_of_data_epoch:
            return False

        if n_data_epochs is None:
            return False

        if self._data_epoch_nr < after_n_data_epochs:
            return False

        return self._data_epoch_nr % n_data_epochs == 0
