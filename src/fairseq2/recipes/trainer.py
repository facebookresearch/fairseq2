# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from enum import Enum
from itertools import count
from statistics import mean
from typing import Generic, TypeVar, final

import torch
import torch.distributed
from torch import Tensor
from torch.optim import Optimizer
from torch.profiler import record_function
from torcheval.metrics import Mean

from fairseq2.checkpoint import CheckpointManager
from fairseq2.datasets import DataReader
from fairseq2.device import DeviceStatTracker
from fairseq2.error import ContractError, InternalError, InvalidOperationError
from fairseq2.gang import Gangs, broadcast_flag
from fairseq2.logging import log
from fairseq2.metrics import MetricBag, MetricDescriptor
from fairseq2.metrics.recorders import MetricRecorder
from fairseq2.nn.utils.gradient import (
    check_gradient_norms,
    normalize_gradients,
)
from fairseq2.optim import DynamicLossScaler
from fairseq2.optim.lr_scheduler import LRScheduler, get_effective_lr
from fairseq2.profilers import Profiler
from fairseq2.recipes.early_stopper import EarlyStopper, NoopEarlyStopper
from fairseq2.recipes.evaluator import EvalUnit
from fairseq2.recipes.metrics import extend_batch_metrics
from fairseq2.recipes.model import Model
from fairseq2.recipes.utils.progress import (
    NoopProgressReporter,
    ProgressReporter,
    ProgressTask,
)
from fairseq2.typing import CPU, ContextManager, DataType
from fairseq2.utils.gc import GarbageCollector
from fairseq2.utils.rng import RngBag
from fairseq2.utils.state import StatefulObjectBag
from fairseq2.utils.stopwatch import Stopwatch

BatchT_contra = TypeVar("BatchT_contra", contravariant=True)


class TrainUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Trainer`."""

    @abstractmethod
    def __call__(self, batch: BatchT_contra) -> tuple[Tensor, int | None]:
        """Process ``batch``.

        :returns:
            - The loss.
            - The number of targets used to compute the loss. If ``None``, the
              model gradients won't be normalized.
        """

    def set_step_nr(self, step_nr: int) -> None:
        """Set the current training step number."""
        pass

    @property
    @abstractmethod
    def model(self) -> Model:
        """The underlying model."""

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag:
        """The training-related metrics."""


class _TrainerState(Enum):
    NOT_STARTED = 0
    RUNNING = 1
    FINISHING = 2
    FINISHED = 3


BatchT = TypeVar("BatchT")


@final
class Trainer(StatefulObjectBag, Generic[BatchT]):
    """Trains a machine learning model."""

    _state: _TrainerState
    _model: Model
    _unit: TrainUnit[BatchT]
    _data_reader: DataReader[BatchT]
    _gangs: Gangs
    _dtype: DataType
    _amp: bool
    _optimizer: Optimizer
    _lr_scheduler: LRScheduler
    _loss_scaler: DynamicLossScaler
    _max_gradient_norm: float | None
    _step_nr: int
    _max_num_steps: int | None
    _data_epoch_nr: int
    _max_num_data_epochs: int | None
    _repeat_step: bool
    _has_read_any_data: bool
    _num_effective_batches: int
    _score_metric_descriptor: MetricDescriptor | None
    _lower_better: bool
    _early_stopper: EarlyStopper | None
    _best_step_and_score: tuple[int, float] | None
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
    _metric_recorder: MetricRecorder
    _publish_metrics_after_n_steps: int
    _publish_metrics_every_n_steps: int | None
    _publish_metrics_after_n_data_epochs: int
    _publish_metrics_every_n_data_epochs: int | None
    _garbage_collector: GarbageCollector
    _profiler: Profiler
    _device_stat_tracker: DeviceStatTracker
    _gradient_check: bool
    _anomaly_detection: bool
    _seed: int
    _rng_bag: RngBag
    _wall_watch: Stopwatch
    _data_read_time: float
    _elapsed_time: float
    _progress_reporter: ProgressReporter
    _progress_task: ProgressTask | None

    def __init__(
        self,
        *,
        unit: TrainUnit[BatchT],
        data_reader: DataReader[BatchT],
        gangs: Gangs,
        dtype: DataType,
        amp: bool,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        checkpoint_manager: CheckpointManager,
        metric_recorder: MetricRecorder,
        garbage_collector: GarbageCollector,
        profiler: Profiler,
        device_stat_tracker: DeviceStatTracker,
        seed: int,
        wall_watch: Stopwatch,
        fp16_loss_scale: tuple[float, float] = (128.0, 0.0001),
        max_gradient_norm: float | None = None,
        max_num_steps: int | None = None,
        max_num_data_epochs: int | None = None,
        score_metric_descriptor: MetricDescriptor | None = None,
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
        publish_metrics_after_n_steps: int = 0,
        publish_metrics_every_n_steps: int | None = None,
        publish_metrics_after_n_data_epochs: int = 0,
        publish_metrics_every_n_data_epochs: int | None = None,
        gradient_check: bool = False,
        anomaly_detection: bool = False,
    ) -> None:
        """
        :param unit:
            The training unit.
        :param data_reader:
            The data reader for training.
        :param gangs:
            The gangs to train on.
        :param optimizer:
            The parameter optimizer.
        :param checkpoint_manager:
            The checkpoint manager.
        :param wall_watch:
            The stopwatch to track process wall-time.
        :param dtype:
            The data type of the model.
        :param lr_scheduler:
            The learning rate scheduler.
        :param amp:
            If ``True``, enables ``torch.amp``.
        :param fp16_loss_scale:
            The initial and minimum loss scale for fp16 training.
        :param max_gradient_norm:
            The maximum gradient norm. If ``None``, no clipping will be applied.
        :param max_num_steps:
            The maximum number of steps to train for.
        :param max_num_data_epochs:
            The maximum number of data epochs to train for.
        :param score_metric_descriptor:
            The descriptor of the metric to use for score calculation.
        :param lower_better:
            If ``True``, a lower metric value is considered a better score.
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
        :param metric_recorder:
            The metric recorder.
        :param publish_metrics_after_n_steps:
            The number of steps after which to start publishing metrics.
        :param publish_metrics_every_n_steps:
            The step interval at which to publish metrics.
        :param publish_metrics_after_n_data_epochs:
            The number of data epochs after which to start publishing metrics.
        :param publish_metrics_every_n_data_epochs:
            The data epoch interval at which to publish metrics.
        :param profile: The profiler.
        :param anomaly_detection:
            If ``True``, turns on anomaly detection feature in ``torch.autograd``.
        :param seed:
            The random number generator seed.
        """
        super().__init__(explicit_only=True)

        self._state = _TrainerState.NOT_STARTED

        device = gangs.root.device

        self._model = unit.model

        self._unit = unit

        self._data_reader = data_reader

        self._gangs = gangs

        self._dtype = dtype

        self._amp = amp

        self._optimizer = optimizer

        self.register_stateful("_lr_scheduler", lr_scheduler)

        fp16_init_scale, fp16_min_scale = fp16_loss_scale

        loss_scaler = DynamicLossScaler(
            optimizer,
            gangs.root,
            sharded=gangs.root.size != gangs.rdp.size,
            init_scale=fp16_init_scale,
            min_scale=fp16_min_scale,
            gradient_accumulation=data_reader.num_accumulate,
            enabled=dtype == torch.float16,
        )

        self.register_stateful("_loss_scaler", loss_scaler)

        self._max_gradient_norm = max_gradient_norm

        self.register_stateful("_step_nr", 0)

        if max_num_steps is not None:
            if max_num_steps <= 0:
                raise ValueError("`max_num_steps` must be greater than zero.")

        self._max_num_steps = max_num_steps

        self.register_stateful("_data_epoch_nr", 1)

        if max_num_data_epochs is not None:
            if max_num_data_epochs <= 0:
                raise ValueError("`max_num_data_epochs` must be greater than zero.")

        self._max_num_data_epochs = max_num_data_epochs

        self._repeat_step = False

        self.register_stateful("_has_read_any_data", False)

        self._num_effective_batches = 0

        self._score_metric_descriptor = score_metric_descriptor

        self._lower_better = lower_better

        if early_stopper is not None:
            if score_metric_descriptor is None:
                raise ValueError(
                    "`score_metric_descriptor` must be specified when `early_stopper` is specified."
                )

            if gangs.root.rank == 0:
                self._early_stopper = early_stopper
            else:
                self._early_stopper = NoopEarlyStopper()
        else:
            self._early_stopper = None

        self.register_stateful("_best_step_and_score", (-1, float("-inf"))

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

        if validate_every_n_steps is not None:
            if validate_every_n_steps <= 0:
                raise ValueError("`validate_every_n_steps` must be greater than zero.")

        self._validate_after_n_steps = validate_after_n_steps
        self._validate_every_n_steps = validate_every_n_steps

        if validate_every_n_data_epochs is not None:
            if validate_every_n_data_epochs <= 0:
                raise ValueError(
                    "`validate_every_n_data_epochs` must be greater than zero."
                )

        self._validate_after_n_data_epochs = validate_after_n_data_epochs
        self._validate_every_n_data_epochs = validate_every_n_data_epochs

        self._checkpoint_manager = checkpoint_manager

        if checkpoint_every_n_steps is not None:
            if checkpoint_every_n_steps <= 0:
                raise ValueError(
                    "`checkpoint_every_n_steps` must be greater than zero."
                )

        self._checkpoint_after_n_steps = checkpoint_after_n_steps
        self._checkpoint_every_n_steps = checkpoint_every_n_steps

        if checkpoint_every_n_data_epochs is not None:
            if checkpoint_every_n_data_epochs <= 0:
                raise ValueError(
                    "`checkpoint_every_n_data_epochs` must be greater than zero."
                )

        self._checkpoint_after_n_data_epochs = checkpoint_after_n_data_epochs
        self._checkpoint_every_n_data_epochs = checkpoint_every_n_data_epochs

        if keep_last_n_checkpoints is not None:
            if keep_best_n_checkpoints is not None:
                raise ValueError(
                    "`keep_last_n_checkpoints` and `keep_best_n_checkpoints` must not be specified at the same time."
                )

            if keep_last_n_checkpoints <= 0:
                raise ValueError("`keep_last_n_checkpoints` must be greater than zero.")
        elif keep_best_n_checkpoints is not None:
            if keep_best_n_checkpoints <= 0:
                raise ValueError("`keep_best_n_checkpoints` must be greater than zero.")

            if checkpoint_every_n_steps is not None:
                if score_metric_descriptor is None:
                    raise ValueError(
                        "`score_metric_descriptor` must be specified when `keep_best_n_checkpoints` is specified."
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
                    "`keep_last_n_models` must not be specified when `keep_last_n_checkpoints` is not specified."
                )

            if keep_last_n_checkpoints > keep_last_n_models:
                raise ValueError(
                    f"`keep_last_n_models` must be greater than or equal to `keep_last_n_checkpoints` ({keep_last_n_checkpoints}), but is {keep_last_n_models} instead."
                )

        if keep_best_n_models is not None:
            if keep_best_n_checkpoints is None:
                raise ValueError(
                    "`keep_best_n_models` must not be specified when `keep_best_n_checkpoints` is not specified."
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

        self.register_stateful("_metric_bag", unit.metric_bag)

        self._metric_recorder = metric_recorder

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

        self._garbage_collector = garbage_collector

        self._profiler = profiler

        self._device_stat_tracker = device_stat_tracker

        self._gradient_check = gradient_check

        self._anomaly_detection = anomaly_detection

        self._seed = seed

        rng_bag = RngBag.from_device_defaults(CPU, device)

        self.register_stateful("_rng_bag", rng_bag)

        self._wall_watch = wall_watch

        self._data_read_time = 0.0

        self._elapsed_time = 0.0

        self._progress_reporter = NoopProgressReporter()

        self._progress_task = None

    def request_stop(self) -> None:
        """Request a graceful stop of the training."""
        if self._state != _TrainerState.RUNNING:
            return

        log.info("Stop requested. Training will be finished.")

        self._state = _TrainerState.FINISHING

    def __call__(self, progress_reporter: ProgressReporter | None = None) -> None:
        if self._state != _TrainerState.NOT_STARTED:
            raise InvalidOperationError("The trainer can only be run once.")

        self._state = _TrainerState.RUNNING

        if progress_reporter is not None:
            self._progress_reporter = progress_reporter

        self._rng_bag.manual_seed(self._seed + self._gangs.root.rank)

        try:
            self._maybe_restore_state()
        except KeyboardInterrupt:
            log.info("Training terminated!")

            raise

        log.info("Running training on {} device(s).", self._gangs.root.size)

        try:
            self._do_run()
        except KeyboardInterrupt:
            log.info("Training terminated at step {}!", self._step_nr)

            raise
        finally:
            self._garbage_collector.enable(False)

        self._gangs.close()

        if self._state != _TrainerState.FINISHED:
            raise InternalError(
                f"`_state` must be `FINISHED`, but is `{self._state}` instead."
            )

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Training finished in {:,} seconds after {} step(s)!", int(elapsed_time), self._step_nr)  # fmt: skip

    def _maybe_restore_state(self) -> None:
        step_nr = self._checkpoint_manager.get_last_step_number()
        if step_nr is None:
            return

        self._step_nr = step_nr

        log.info("Restoring training from the last checkpoint at step {}.", step_nr)  # fmt: skip

        log.info("Restoring the trainer state.")

        self._checkpoint_manager.load_trainer_state(step_nr, self)

        log.info("Trainer state restored.")

        log.info("Restoring the optimizer state.")

        self._checkpoint_manager.load_optimizer_state(step_nr, self._optimizer)

        log.info("Optimizer state restored.")

        log.info("Restoring the data reader state.")

        self._checkpoint_manager.load_data_reader_state(step_nr, self._data_reader)

        log.info("Data reader state restored.")

        self._gangs.root.barrier()

        log.info("Training restored. Resuming.")

        if self._max_num_steps is not None:
            if self._step_nr >= self._max_num_steps:
                self._state = _TrainerState.FINISHING

    def _do_run(self) -> None:
        self._model.module.train()

        self._garbage_collector.enable()

        with self._progress_reporter, self._profiler:
            self._progress_task = self._progress_reporter.create_task(
                "train", total=self._max_num_steps, completed=self._step_nr
            )

            self._device_stat_tracker.reset()

            first_iter = True

            while self._state == _TrainerState.RUNNING:
                self._step_nr += 1

                self._progress_task.step(1)

                self._checkpoint_manager.maybe_complete_async_checkpoint()

                detect_anomaly = torch.autograd.set_detect_anomaly(  # type: ignore[attr-defined]
                    self._anomaly_detection, check_nan=True
                )

                with detect_anomaly:
                    with record_function(f"step_{self._step_nr}"):
                        self._run_step()

                if self._repeat_step:
                    self._step_nr -= 1

                    self._progress_task.step(-1)
                elif self._max_num_steps is not None:
                    if self._step_nr >= self._max_num_steps:
                        self._state = _TrainerState.FINISHING

                if self._should_publish_metrics():
                    self._publish_metrics()

                if self._should_checkpoint():
                    self._checkpoint()

                if self._should_validate():
                    self._validate()

                self._profiler.step()

                self._garbage_collector.step()

                if first_iter:
                    # Emptying the CUDA memory allocator cache after the first
                    # iteration can reduce fragmentation and avoid OOM.
                    if self._gangs.root.device.type == "cuda":
                        torch.cuda.empty_cache()

                    first_iter = False

        if self._state != _TrainerState.FINISHING:
            raise InternalError(
                f"`_state` must be `FINISHING`, but is `{self._state}` instead."
            )

        self._state = _TrainerState.FINISHED

    def _run_step(self) -> None:
        step_nr = self._step_nr

        log.debug("{} training step {}.", "Repeating" if self._repeat_step else "Running", step_nr)  # fmt: skip

        watch = Stopwatch(start=True, device=self._gangs.root.device)

        # Prepare the unit.
        if not self._repeat_step:
            with record_function(f"step_{step_nr}_unit_setup"):
                self._unit.set_step_nr(step_nr)

        # Collect the batches.
        with record_function(f"step_{step_nr}_data_load"):
            batches = self._next_batches()
            if batches is None:
                return

        num_targets = 0

        if self._loss_scaler.is_enabled:
            self._metric_bag.begin_updates()

        # Accumulate.
        for batch_nr, batch in enumerate(batches):
            with self._maybe_no_sync(batch_nr, len(batches)):
                with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                    batch_loss, num_batch_targets = self._compute_loss(batch)

                if num_batch_targets is not None:
                    if num_batch_targets == 0:
                        raise ContractError(
                            "The train unit returned zero loss targets."
                        )

                    num_targets += num_batch_targets

                with record_function(f"step_{step_nr}_{batch_nr}_backward"):
                    self._loss_scaler.backward(batch_loss)

        # Normalize.
        if num_targets > 0:
            normalize_gradients(
                self._model.module, self._gangs.dp, num_targets=num_targets
            )

        # Clip.
        with record_function(f"step_{step_nr}_grad_norm"):
            self._loss_scaler.unscale_gradients_()

            # TODO(balioglu): Support tensor parallelism!
            grad_norm = self._model.clip_gradient_norm(self._max_gradient_norm)

            if self._gradient_check:
                # Sanity check.
                if not check_gradient_norms(grad_norm, self._gangs.dp, step_nr):
                    raise FloatingPointError(
                        f"The gradients are inconsistent between processes at step {step_nr}. Training cannot continue."
                    )

        # Update the parameters.
        with record_function(f"step_{step_nr}_optimizer"):
            _, scale_result = self._loss_scaler.run_optimizer_step(step_nr)

        self._repeat_step = scale_result.overflow

        if self._repeat_step:
            if scale_result.min_reached:
                raise FloatingPointError(
                    f"The gradients are scaled down to minimum at step {step_nr}. Training cannot continue."
                )

            self._metric_bag.rollback_updates()
        else:
            self._lr_scheduler.step()

            if self._loss_scaler.is_enabled:
                self._metric_bag.commit_updates()

            self._metric_bag.gradient_norm.update(grad_norm)

            self._num_effective_batches += 1

        # Reset the grads.
        self._optimizer.zero_grad(set_to_none=True)

        self._elapsed_time += watch.get_elapsed_time()

    def _next_batches(self) -> list[BatchT] | None:
        if self._state != _TrainerState.RUNNING:
            raise InternalError(
                f"`_state` must be `RUNNING`, but is `{self._state}` instead."
            )

        watch = Stopwatch(start=True)

        try:
            batches = next(self._data_reader)
        except StopIteration:
            batches = None

        self._data_read_time += watch.get_elapsed_time()

        if batches is not None:
            self._has_read_any_data = True

            return batches

        self._data_reader.reset()

        log.info("End of epoch {} reached at training step {}.", self._data_epoch_nr, self._step_nr)  # fmt: skip

        if not self._has_read_any_data:
            self._state = _TrainerState.FINISHING
        elif self._max_num_data_epochs is not None:
            if self._data_epoch_nr >= self._max_num_data_epochs:
                self._state = _TrainerState.FINISHING

        if self._state == _TrainerState.FINISHING:
            log.info("End of data reached.", self._step_nr)
        else:
            self._data_epoch_nr += 1

            self._repeat_step = True

        return None

    def _maybe_no_sync(self, batch_nr: int, num_batches: int) -> ContextManager:
        if batch_nr < num_batches - 1:
            return self._model.no_sync()

        return nullcontext()

    def _compute_loss(self, batch: BatchT) -> tuple[Tensor, int | None]:
        with self._maybe_autocast():
            return self._unit(batch)

    def _maybe_autocast(self) -> ContextManager:
        if self._dtype == torch.float32 or not self._amp:
            return nullcontext()

        return torch.autocast(device_type=self._gangs.dp.device.type, dtype=self._dtype)

    def _should_publish_metrics(self) -> bool:
        return self._should_do(
            self._publish_metrics_after_n_steps,
            self._publish_metrics_every_n_steps,
            self._publish_metrics_after_n_data_epochs,
            self._publish_metrics_every_n_data_epochs,
        )

    def _publish_metrics(self) -> None:
        log.debug("Syncing metrics.")

        if self._gangs.tp.rank == 0:
            values = self._metric_bag.sync_and_compute_metrics()
        else:
            values = None

        self._metric_bag.reset_non_persistent_metrics()

        if self._gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            extend_batch_metrics(
                values, self._num_effective_batches, self._elapsed_time
            )

            device_stats = self._device_stat_tracker.get_stats()

            values.update(device_stats)

            values["lr"] = get_effective_lr(self._lr_scheduler)

            values["data_epoch"] = self._data_epoch_nr

            values["data_read_time"] = self._data_read_time

            values["elapsed_time"] = self._elapsed_time

            values["wall_time"] = self._wall_watch.get_elapsed_time()

            self._metric_recorder.record_metrics("train", values, self._step_nr)

        self._num_effective_batches = 0

        self._data_read_time = 0.0

        self._elapsed_time = 0.0

        self._device_stat_tracker.reset()

        self._gangs.root.barrier()

    def _should_validate(self) -> bool:
        if not self._valid_units:
            return False

        return self._should_do(
            self._validate_after_n_steps,
            self._validate_every_n_steps,
            self._validate_after_n_data_epochs,
            self._validate_every_n_data_epochs,
        )

    def _validate(self) -> None:
        log.info("Starting validation after step {}.", self._step_nr)

        self._model.module.eval()

        with self._model.summon_full_parameters():
            unit_scores = []

            for unit, data_reader in zip(self._valid_units, self._valid_data_readers):
                if unit.name:
                    log.info("Validating {}.", unit.name)

                unit_score = self._validate_unit(unit, data_reader)
                if unit_score is not None:
                    unit_scores.append(unit_score)

        score = self._compute_score(unit_scores)

        if self._should_checkpoint():
            self._checkpoint_manager.save_score(self._step_nr, score)

        self._maybe_request_early_stop(score)

        self._model.module.train()

        log.info("Validation complete.")

    @torch.inference_mode()
    def _validate_unit(
        self, unit: EvalUnit[BatchT], data_reader: DataReader[BatchT]
    ) -> float | None:
        watch = Stopwatch(start=True, device=self._gangs.root.device)

        unit.set_step_nr(self._step_nr)

        task = self._progress_reporter.create_task("valid", total=None)

        num_effective_batches = 0

        for step_nr in count(start=1):
            task.step(1)

            log.debug("Running validation step {}.", step_nr)

            try:
                batches = next(data_reader)
            except StopIteration:
                break

            for batch in batches:
                with self._maybe_autocast():
                    unit(batch)

            num_effective_batches += 1

        task.close()

        data_reader.reset()

        metric_values = self._publish_validation_metrics(
            unit, num_effective_batches, watch.get_elapsed_time()
        )

        return self._get_unit_score(metric_values)

    def _publish_validation_metrics(
        self, unit: EvalUnit[BatchT], num_batches: int, elapsed_time: float
    ) -> dict[str, object] | None:
        gangs = self._gangs

        log.debug("Syncing validation metrics.")

        if gangs.tp.rank == 0:
            values = unit.metric_bag.sync_and_compute_metrics()
        else:
            values = None

        unit.metric_bag.reset_metrics()

        if gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            extend_batch_metrics(values, num_batches, elapsed_time)

            values["data_epoch"] = self._data_epoch_nr

            values["elapsed_time"] = elapsed_time

            values["wall_time"] = self._wall_watch.get_elapsed_time()

            run = "valid"

            if unit.name:
                run = f"{run}/{unit.name}"

            self._metric_recorder.record_metrics(run, values, self._step_nr)

        gangs.root.barrier()

        return values

    def _get_unit_score(self, metric_values: dict[str, object] | None) -> float | None:
        if metric_values is None:
            return None

        if self._score_metric_descriptor is None:
            return None

        metric_value = metric_values.get(self._score_metric_descriptor.name)
        if metric_score is None:
            return None

        if not isinstance(metric_score, (int, float, Tensor)):
            log.warning("The score metric must be of type `int`, `float`, or `torch.Tensor`.")  # fmt: skip

            return None

        score = float(metric_value)

        if self._lower_better:
            score = -score

        return score

    def _compute_score(self, unit_scores: list[float]) -> float | None:
        if self._score_metric_descriptor is None:
            return None

        if not unit_scores:
            if self._gangs.root.rank == 0:
                raise ContractError(
                    "None of the validation units returned a score metric value."
                )

            return None

        score = mean(unit_scores)

        if score > self._best_step_and_score[1]:
            self._best_step_and_score = (self._step_nr, score)

        if log.is_enabled_for_info():
            best_step_nr, best_score = self._best_step_and_score

            if len(unit_scores) > 1:
                m1 = "Mean "
                m2 = "Best Mean "
            else:
                m1 = ""
                m2 = "Best "

            v1 = self._score_metric_descriptor.formatter(score)
            v2 = self._score_metric_descriptor.formatter(best_score)

            s1 = f"{self._score_metric_descriptor.display_name}: {v1}"
            s2 = f"{self._score_metric_descriptor.display_name}: {v2}"

            log.info("Score (step {}) - {}{} | {}{} at step {}", self._step_nr, m1, s1, m2, s2, best_step_nr)  # fmt: skip

        return last_score

    def _maybe_request_early_stop(self, score: float | None) -> None:
        if self._early_stopper is None:
            return

        if self._state != _TrainerState.RUNNING:
            return

        if self._gangs.root.rank == 0:
            if score is None:
                raise InternalError("Early stopping, but `score` is `None`.")

            should_stop = self._early_stopper.should_stop(self._step_nr, score)
        else:
            should_stop = False

        should_stop = broadcast_flag(self._gangs.root, should_stop)
        if should_stop:
            log.info("Early stop requested. Training will be finished.")

            self._state = _TrainerState.FINISHING

    def _should_checkpoint(self) -> bool:
        return self._should_do(
            self._checkpoint_after_n_steps,
            self._checkpoint_every_n_steps,
            self._checkpoint_after_n_data_epochs,
            self._checkpoint_every_n_data_epochs,
        )

    def _checkpoint(self) -> None:
        watch = Stopwatch(start=True, device=self._gangs.root.device)

        step_nr = self._step_nr

        if self._checkpoint_manager.is_saving():
            log.info("Waiting the current asynchronous checkpoint operation to complete before saving the checkpoint at step {}.", step_nr)  # fmt: skip

            self._checkpoint_manager.maybe_complete_async_checkpoint(block=True)

        block = self._state == _TrainerState.FINISHING

        if block:
            log.info("Saving the final checkpoint at step {}.", step_nr)
        else:
            log.info("Asynchronously saving the checkpoint at step {}.", step_nr)

        self._checkpoint_manager.save_checkpoint(
            step_nr,
            self,
            self._model,
            self._optimizer,
            self._data_reader,
            callback=self._complete_checkpoint,
            block=block,
        )

        self._elapsed_time += watch.get_elapsed_time()

    def _complete_checkpoint(self, step_nr: int) -> None:
        # Clean up the checkpoints.
        nc = self._keep_last_n_checkpoints
        nm = self._keep_last_n_models

        if nm is not None:
            if nc is None:
                raise InternalError("`_keep_last_n_checkpoints` is `None`")

            self._checkpoint_manager.keep_last_n_checkpoints(nm)
            self._checkpoint_manager.keep_last_n_checkpoints(nc, preserve_model=True)
        elif nc is not None:
            self._checkpoint_manager.keep_last_n_checkpoints(nc)

        nc = self._keep_best_n_checkpoints
        nm = self._keep_best_n_models

        if nm is not None:
            if nc is None:
                raise InternalError("`_keep_best_n_checkpoints` is `None`")

            self._checkpoint_manager.keep_best_n_checkpoints(nm)
            self._checkpoint_manager.keep_best_n_checkpoints(nc, preserve_model=True)
        elif nc is not None:
            self._checkpoint_manager.keep_best_n_checkpoints(nc)

        log.info("Checkpoint at step {} saved.", step_nr)

    def _should_do(
        self,
        after_n_steps: int,
        every_n_steps: int | None,
        after_n_data_epochs: int,
        every_n_data_epochs: int | None,
    ) -> bool:
        if self._state == _TrainerState.FINISHING:
            return True

        if self._state == _TrainerState.RUNNING:
            if self._repeat_step:
                return False

            if every_n_steps is not None:
                if self._step_nr >= after_n_steps:
                    if self._step_nr % every_n_steps == 0:
                        return True

            if every_n_data_epochs is not None:
                if self._data_epoch_nr >= after_n_data_epochs:
                    if self._data_epoch_nr % every_n_data_epochs == 0:
                        return True

        return False
