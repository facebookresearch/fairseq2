# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from enum import Enum
from typing import Final, Generic, Mapping, TypeVar, final

import torch
import torch.distributed
from torch import Tensor
from torch.optim import Optimizer
from torch.profiler import record_function
from typing_extensions import override

from fairseq2.checkpoint import (
    CheckpointError,
    CheckpointLoadError,
    CheckpointManager,
    CheckpointSaveError,
    CheckpointState,
)
from fairseq2.datasets import DataReader, DataReadError
from fairseq2.device import DeviceStatTracker
from fairseq2.error import InternalError, InvalidOperationError
from fairseq2.gang import GangError, Gangs, broadcast_flag
from fairseq2.logging import log
from fairseq2.metrics import Mean, MetricBag, MetricBagError, MetricDescriptor
from fairseq2.metrics.recorders import MetricRecorder, MetricRecordError
from fairseq2.nn.utils.gradient import check_gradient_norms, normalize_gradients
from fairseq2.optim import DynamicLossScaler
from fairseq2.optim.lr_scheduler import LRScheduler, get_effective_lr
from fairseq2.profilers import Profiler
from fairseq2.recipes._early_stopper import EarlyStopper
from fairseq2.recipes._error import (
    InconsistentGradientNormError,
    MinimumLossScaleReachedError,
    RecipeError,
    UnitError,
)
from fairseq2.recipes._metrics import extend_batch_metrics
from fairseq2.recipes._model import Model
from fairseq2.recipes._recipe import Recipe, RecipeStopException
from fairseq2.recipes._validator import Validator
from fairseq2.typing import CPU, ContextManager, DataType
from fairseq2.utils.gc import GarbageCollector
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rng import RngBag
from fairseq2.utils.state import Stateful
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
        pass

    @property
    @abstractmethod
    def model(self) -> Model: ...

    @property
    @abstractmethod
    def metric_bag(self) -> MetricBag: ...


BatchT = TypeVar("BatchT")


@final
class Trainer(Recipe, Generic[BatchT]):
    """Trains a machine learning model."""

    _state: TrainerState
    _step_nr: int
    _data_epoch_nr: int
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
    _gradient_check: bool
    _anomaly_detection: bool
    _rng_bag: RngBag
    _seed: int
    _max_num_steps: int | None
    _max_num_data_epochs: int | None
    _validator: Validator
    _validate_after_n_steps: int
    _validate_every_n_steps: int | None
    _validate_after_n_data_epochs: int
    _validate_every_n_data_epochs: int | None
    _early_stopper: EarlyStopper | None
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
    _data_watch: Stopwatch
    _compute_watch: Stopwatch
    _lapse_watch: Stopwatch
    _wall_watch: Stopwatch
    _base_wall_time: float
    _progress_reporter: ProgressReporter
    _stop_requested: bool
    _repeat_step: bool
    _has_read_any_data: bool
    _num_batches_read: int
    _last_lr: float

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
        seed: int,
        validator: Validator,
        checkpoint_manager: CheckpointManager,
        metric_recorder: MetricRecorder,
        garbage_collector: GarbageCollector,
        profiler: Profiler,
        device_stat_tracker: DeviceStatTracker,
        wall_watch: Stopwatch,
        progress_reporter: ProgressReporter,
        fp16_loss_scale: tuple[float, float] = (128.0, 0.0001),
        max_gradient_norm: float | None = None,
        gradient_check: bool = False,
        anomaly_detection: bool = False,
        max_num_steps: int | None = None,
        max_num_data_epochs: int | None = None,
        validate_after_n_steps: int = 0,
        validate_every_n_steps: int | None = None,
        validate_after_n_data_epochs: int = 0,
        validate_every_n_data_epochs: int | None = None,
        early_stopper: EarlyStopper | None = None,
        score_metric_descriptor: MetricDescriptor | None = None,
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
        :param validate_after_n_steps:
            The number of steps after which to start validating the model.
        :param validate_every_n_steps:
            The step interval at which to validate the model.
        :param validate_after_n_data_epochs:
            The number of data epochs after which to start validating the model.
        :param validate_every_n_data_epochs:
            The data epoch interval at which to validate the model.
        :param early_stopper:
            The early-stopper callable.
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
        self._state = TrainerState.NOT_STARTED

        self._step_nr = 0

        self._data_epoch_nr = 1

        self._model = unit.model

        self._unit = unit

        self._data_reader = data_reader

        self._gangs = gangs

        self._dtype = dtype

        self._amp = amp

        self._optimizer = optimizer

        self._lr_scheduler = lr_scheduler

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

        self._loss_scaler = loss_scaler

        self._max_gradient_norm = max_gradient_norm

        self._gradient_check = gradient_check

        self._anomaly_detection = anomaly_detection

        self._rng_bag = RngBag.from_device_defaults(CPU, gangs.root.device)

        self._seed = seed

        if max_num_steps is not None:
            if max_num_steps <= 0:
                raise ValueError("`max_num_steps` must be greater than zero.")

        self._max_num_steps = max_num_steps

        if max_num_data_epochs is not None:
            if max_num_data_epochs <= 0:
                raise ValueError("`max_num_data_epochs` must be greater than zero.")

        self._max_num_data_epochs = max_num_data_epochs

        self._validator = validator

        if validate_every_n_steps is not None:
            if validate_every_n_steps <= 0:
                raise ValueError("`validate_every_n_steps` must be greater than zero.")

            if publish_metrics_every_n_steps is not None:
                if validate_every_n_steps % publish_metrics_every_n_steps != 0:
                    raise ValueError(
                        f"`validate_every_n_steps` must be a multiple of `publish_metrics_every_n_steps` ({publish_metrics_every_n_steps}), but is {validate_every_n_steps} instead."
                    )

        self._validate_after_n_steps = validate_after_n_steps
        self._validate_every_n_steps = validate_every_n_steps

        if validate_every_n_data_epochs is not None:
            if validate_every_n_data_epochs <= 0:
                raise ValueError(
                    "`validate_every_n_data_epochs` must be greater than zero."
                )

            if publish_metrics_every_n_data_epochs is not None:
                if validate_every_n_data_epochs % publish_metrics_every_n_data_epochs != 0:  # fmt: skip
                    raise ValueError(
                        f"`validate_every_n_data_epochs` must be a multiple of `publish_metrics_every_n_data_epochs` ({publish_metrics_every_n_data_epochs}), but is {validate_every_n_data_epochs} instead."
                    )

        self._validate_after_n_data_epochs = validate_after_n_data_epochs
        self._validate_every_n_data_epochs = validate_every_n_data_epochs

        self._early_stopper = early_stopper

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
            "gradient_norm", Mean(device=gangs.root.device), persistent=False
        )

        self._metric_bag = unit.metric_bag

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

        self._data_watch = Stopwatch()

        self._compute_watch = Stopwatch(device=gangs.root.device)

        self._lapse_watch = Stopwatch()

        self._wall_watch = wall_watch

        self._base_wall_time = 0.0

        self._progress_reporter = progress_reporter

        self._stop_requested = False

        self._repeat_step = False

        self._has_read_any_data = False

        self._num_batches_read = 0

        self._last_lr = 0.0

    @override
    def run(self) -> None:
        if self._state != TrainerState.NOT_STARTED:
            raise InvalidOperationError("The trainer cannot be run more than once.")

        self._state = TrainerState.RUNNING

        gangs = self._gangs

        with self._progress_reporter:
            with self._rng_bag.temporary_manual_seed(self._seed + gangs.root.rank):
                self._maybe_restore_state()

                with self._profiler, self._garbage_collector:
                    self._do_run()

        if self._state != TrainerState.STOPPED:
            raise InternalError(
                f"`_state` must be `STOPPED`, but is `{self._state}` instead."
            )

        gangs.close()

        if self._stop_requested:
            raise RecipeStopException()

    def _maybe_restore_state(self) -> None:
        try:
            step_nr = self._checkpoint_manager.maybe_get_last_step_number()
        except CheckpointError as ex:
            raise RecipeError(
                "The checkpoints cannot accessed. See the nested exception for details."
            ) from ex

        if step_nr is None:
            return

        log.info("Restoring training from the last checkpoint at step {}.", step_nr)

        try:
            log.info("Restoring the trainer state.")

            trainer_state_bag = TrainerStateBag(self)

            self._checkpoint_manager.load_trainer_state(step_nr, trainer_state_bag)

            log.info("Trainer state restored.")

            log.info("Restoring the optimizer state.")

            self._checkpoint_manager.load_optimizer_state(step_nr, self._optimizer)

            log.info("Optimizer state restored.")

            log.info("Restoring the data reader state.")

            self._checkpoint_manager.load_data_reader_state(step_nr, self._data_reader)

            log.info("Data reader state restored.")
        except CheckpointLoadError as ex:
            raise RecipeError(
                f"The last checkpoint at step {ex.step_nr} cannot be loaded. See the nested exception for details."
            ) from ex

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise RecipeError(
                "The collective barrier after the checkpoint load operation has failed. See the nested exception for details."
            ) from ex

        log.info("Training restored. Resuming from step {}.", step_nr)

        if self._max_num_steps is not None:
            if self._step_nr >= self._max_num_steps:
                self._state = TrainerState.STOPPING

        if self._max_num_data_epochs is not None:
            if self._data_epoch_nr >= self._max_num_data_epochs:
                self._state = TrainerState.STOPPING

    def _do_run(self) -> None:
        self._model.module.train()

        progress_task = self._progress_reporter.create_task(
            "train", total=self._max_num_steps, completed=self._step_nr
        )

        self._device_stat_tracker.reset()

        first_iter = True

        with progress_task, self._lapse_watch:
            while self._state == TrainerState.RUNNING:
                self._step_nr += 1

                progress_task.step(1)

                try:
                    self._checkpoint_manager.maybe_complete_async_checkpoint()
                except CheckpointSaveError as ex:
                    raise RecipeError(
                        f"The checkpoint of step {ex.step_nr} cannot be saved. See the nested exception for details."
                    ) from ex

                detect_anomaly = torch.autograd.set_detect_anomaly(  # type: ignore[attr-defined]
                    self._anomaly_detection, check_nan=True
                )

                with detect_anomaly:
                    with record_function(f"step_{self._step_nr}"):
                        self._run_step()

                if self._repeat_step:
                    self._step_nr -= 1

                    progress_task.step(-1)
                elif self._max_num_steps is not None:
                    if self._step_nr >= self._max_num_steps:
                        self._state = TrainerState.STOPPING

                if self._should_checkpoint():
                    self._checkpoint()

                if self._should_publish_metrics():
                    self._publish_metrics()

                    self._metric_bag.reset_non_persistent_metrics()

                    self._reset_watches()

                    self._num_batches_read = 0

                    self._device_stat_tracker.reset()

                if self._should_validate():
                    self._lapse_watch.stop()

                    self._validate()

                    self._lapse_watch.start()

                self._profiler.step()

                self._garbage_collector.step()

                if first_iter:
                    # Emptying the CUDA memory allocator cache after the first
                    # iteration can reduce fragmentation and avoid OOM.
                    if self._gangs.root.device.type == "cuda":
                        torch.cuda.empty_cache()

                    first_iter = False

        if self._state != TrainerState.STOPPING:
            raise InternalError(
                f"`_state` must be `STOPPING`, but is `{self._state}` instead."
            )

        self._state = TrainerState.STOPPED

    def _run_step(self) -> None:
        step_nr = self._step_nr

        log.debug("{} step {}.", "Repeating" if self._repeat_step else "Running", step_nr)  # fmt: skip

        gangs = self._gangs

        # Prepare the unit.
        if not self._repeat_step:
            with self._compute_watch:
                with record_function(f"step_{step_nr}_setup"):
                    try:
                        self._unit.set_step_nr(step_nr)
                    except UnitError as ex:
                        raise RecipeError(
                            "The train unit has failed. See the nested exception for details."
                        ) from ex

        # Collect the batches.
        with self._data_watch:
            with record_function(f"step_{step_nr}_data_load"):
                batches = self._read_next_batches()
                if batches is None:
                    return

        num_targets = 0

        if self._loss_scaler.is_enabled:
            self._metric_bag.begin_updates()

        # Run the model.
        with self._compute_watch:
            for batch_nr, batch in enumerate(batches):
                with self._maybe_no_sync(batch_nr, len(batches)):
                    with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                        loss, num_batch_targets = self._compute_loss(batch)

                    # If the unit does not return the number of logit targets
                    # of this batch, we assume that the loss is the mean loss
                    # and that each batch in this step has the same number of
                    # logit targets. In this case, we don't need to normalize
                    # the gradients at the end of the step, but we still have
                    # to take gradient accumulation into account.
                    if num_batch_targets is None:
                        loss = loss / len(batches)
                    else:
                        num_targets += num_batch_targets

                    with record_function(f"step_{step_nr}_{batch_nr}_backward"):
                        self._loss_scaler.backward(loss)

            # This function gathers the total number of logit targets across all
            # processes and divides the gradients by it. This is needed when the
            # batches have varying sizes and we cannot normalize the loss before
            # the backward pass.
            if num_targets > 0:
                normalize_gradients(self._model.module, gangs.dp, num_targets)

            self._loss_scaler.unscale_gradients_()

            # Clip the gradients.
            with record_function(f"step_{step_nr}_grad_norm"):
                # TODO(balioglu): Support tensor parallelism!
                grad_norm = self._model.clip_gradient_norm(self._max_gradient_norm)

                if self._gradient_check:
                    if not check_gradient_norms(grad_norm, gangs.dp, step_nr):
                        raise InconsistentGradientNormError(step_nr)

            # Update the parameters.
            with record_function(f"step_{step_nr}_optimizer"):
                _, scale_result = self._loss_scaler.run_optimizer_step(step_nr)

            # Reset.
            self._optimizer.zero_grad(set_to_none=True)

        self._repeat_step = scale_result.overflow

        if self._repeat_step:
            if scale_result.min_reached:
                raise MinimumLossScaleReachedError(step_nr)

            self._metric_bag.rollback_updates()

            return

        self._last_lr = get_effective_lr(self._lr_scheduler)

        self._lr_scheduler.step()

        if self._loss_scaler.is_enabled:
            self._metric_bag.commit_updates()

        self._metric_bag.gradient_norm.update(grad_norm)

        self._num_batches_read += 1

    def _read_next_batches(self) -> list[BatchT] | None:
        if self._state != TrainerState.RUNNING:
            raise InternalError(
                f"`_state` must be `RUNNING`, but is `{self._state}` instead."
            )

        try:
            batches = next(self._data_reader)
        except DataReadError as ex:
            raise RecipeError(
                "The train data read operation has failed. See the nested exception for details."
            ) from ex
        except StopIteration:
            batches = None

        if batches is not None:
            self._has_read_any_data = True

            return batches

        self._data_reader.reset()

        log.info("End of epoch {} reached at step {}.", self._data_epoch_nr, self._step_nr)  # fmt: skip

        end_of_data = False

        if not self._has_read_any_data:
            end_of_data = True
        elif self._max_num_data_epochs is not None:
            if self._data_epoch_nr >= self._max_num_data_epochs:
                end_of_data = True

        if end_of_data:
            self._state = TrainerState.STOPPING

            log.info("End of data reached.", self._step_nr)
        else:
            self._data_epoch_nr += 1

        self._repeat_step = not end_of_data

        return None

    def _maybe_no_sync(self, batch_nr: int, num_batches: int) -> ContextManager:
        if batch_nr < num_batches - 1:
            return self._model.no_sync()

        return nullcontext()

    def _compute_loss(self, batch: BatchT) -> tuple[Tensor, int | None]:
        with self._maybe_autocast():
            try:
                return self._unit(batch)
            except UnitError as ex:
                raise RecipeError(
                    "The train unit has failed. See the nested exception for details."
                ) from ex

    def _maybe_autocast(self) -> ContextManager:
        if self._dtype == torch.float32 or not self._amp:
            return nullcontext()

        device_type = self._gangs.root.device.type

        return torch.autocast(device_type=device_type, dtype=self._dtype)

    def _should_publish_metrics(self) -> bool:
        return self._should_do(
            self._publish_metrics_after_n_steps,
            self._publish_metrics_every_n_steps,
            self._publish_metrics_after_n_data_epochs,
            self._publish_metrics_every_n_data_epochs,
        )

    def _publish_metrics(self) -> None:
        log.debug("Syncing train metrics.")

        gangs = self._gangs

        try:
            if gangs.tp.rank == 0:
                values = self._metric_bag.sync_and_compute_metrics()
            else:
                values = None
        except MetricBagError as ex:
            raise RecipeError(
                "The train metric values cannot be synced across processes. See the nested exception for details."
            ) from ex

        if gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            values["lr"] = self._last_lr

            values["data_epoch"] = self._data_epoch_nr

            device_stats = self._device_stat_tracker.get_stats()

            values.update(device_stats)

            data_time = self._data_watch.get_elapsed_time()

            compute_time = self._compute_watch.get_elapsed_time()

            extend_batch_metrics(
                values, self._num_batches_read, data_time + compute_time
            )

            values["data_time"] = data_time

            values["compute_time"] = compute_time

            values["lapse_time"] = self._lapse_watch.get_elapsed_time()

            wall_time = self._wall_watch.get_elapsed_time()

            values["total_time"] = self._base_wall_time + wall_time

            values["wall_time"] = wall_time

            try:
                self._metric_recorder.record_metrics("train", values, self._step_nr)
            except MetricRecordError as ex:
                raise RecipeError(
                    f"The train metric values of step {self._step_nr} cannot recorded. See the nested exception for details."
                ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise RecipeError(
                "The collective barrier after the metric sync operation has failed. See the nested exception for details."
            ) from ex

    def _reset_watches(self) -> None:
        self._data_watch.reset()

        self._compute_watch.reset()

        self._lapse_watch.reset()

    def _should_validate(self) -> bool:
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
            score = self._validator.run(self._step_nr)

        if score is not None:
            if self._should_checkpoint():
                self._save_score(score)

            self._maybe_request_early_stop(score)

        self._validator.reset()

        self._model.module.train()

        if self._state == TrainerState.STOPPING:
            log.info("Validation finished.")
        else:
            log.info("Validation finished. Resuming training from step {}.", self._step_nr)  # fmt: skip

    def _save_score(self, score: float) -> None:
        try:
            self._checkpoint_manager.save_score(self._step_nr, score)
        except CheckpointSaveError as ex:
            raise RecipeError(
                f"The score of step {self._step_nr} cannot be saved. See the nested exception for details."
            ) from ex

        num_chkpt = self._keep_best_n_checkpoints
        num_model = self._keep_best_n_models

        if num_chkpt is None and num_model is None:
            return

        if self._checkpoint_manager.is_saving():
            log.info("Waiting for the current checkpoint save operation to complete before deleting the old checkpoints.")  # fmt: skip

            try:
                self._checkpoint_manager.maybe_complete_async_checkpoint(block=True)
            except CheckpointSaveError as ex:
                raise RecipeError(
                    f"The checkpoint of step {ex.step_nr} cannot be saved. See the nested exception for details."
                ) from ex

        try:
            if num_model is not None:
                if num_chkpt is None:
                    raise InternalError("`_keep_best_n_checkpoints` is `None`")

                self._checkpoint_manager.keep_best_n_checkpoints(num_model)

                self._checkpoint_manager.keep_best_n_checkpoints(
                    num_chkpt, preserve_model=True
                )
            elif num_chkpt is not None:
                self._checkpoint_manager.keep_best_n_checkpoints(num_chkpt)
        except CheckpointError as ex:
            raise RecipeError(
                f"The previous checkpoints before step {self._step_nr} cannot be deleted. See the nested exception for details."
            ) from ex

    def _maybe_request_early_stop(self, score: float) -> None:
        if self._early_stopper is None:
            return

        if self._state != TrainerState.RUNNING:
            return

        gangs = self._gangs

        if gangs.root.rank == 0:
            should_stop = self._early_stopper.should_stop(self._step_nr, score)
        else:
            should_stop = False

        should_stop = broadcast_flag(gangs.root, should_stop)
        if should_stop:
            log.info("Early stop requested. Training will be stopped.")

            self._stop_requested = True

            self._state = TrainerState.STOPPING

    def _should_checkpoint(self) -> bool:
        return self._should_do(
            self._checkpoint_after_n_steps,
            self._checkpoint_every_n_steps,
            self._checkpoint_after_n_data_epochs,
            self._checkpoint_every_n_data_epochs,
        )

    def _checkpoint(self) -> None:
        step_nr = self._step_nr

        if self._checkpoint_manager.is_saving():
            log.info("Waiting for the current checkpoint save operation to complete before saving the next checkpoint at step {}.", step_nr)  # fmt: skip

            try:
                self._checkpoint_manager.maybe_complete_async_checkpoint(block=True)
            except CheckpointSaveError as ex:
                raise RecipeError(
                    f"The checkpoint of step {ex.step_nr} cannot be saved. See the nested exception for details."
                ) from ex

        block = self._state == TrainerState.STOPPING

        log.info("Preparing the checkpoint at step {}.", step_nr)

        def log_ready(step_nr: int, state: CheckpointState) -> None:
            if block:
                log.info("Checkpoint prepared. Saving.")
            else:
                log.info("Checkpoint prepared. Saving asynchronously.")

        trainer_state_bag = TrainerStateBag(self)

        tmp = self._base_wall_time

        self._base_wall_time += self._wall_watch.get_elapsed_time()

        try:
            self._checkpoint_manager.save_checkpoint(
                step_nr,
                trainer_state_bag,
                self._model,
                self._optimizer,
                self._data_reader,
                state_processor=log_ready,
                callback=self._complete_checkpoint,
                block=block,
            )
        except CheckpointSaveError as ex:
            raise RecipeError(
                f"The checkpoint of step {ex.step_nr} cannot be saved. See the nested exception for details."
            ) from ex

        self._base_wall_time = tmp

    def _complete_checkpoint(self, step_nr: int) -> None:
        try:
            num_chkpt = self._keep_last_n_checkpoints
            num_model = self._keep_last_n_models

            if num_model is not None:
                if num_chkpt is None:
                    raise InternalError("`_keep_last_n_checkpoints` is `None`")

                self._checkpoint_manager.keep_last_n_checkpoints(num_model)

                self._checkpoint_manager.keep_last_n_checkpoints(
                    num_chkpt, preserve_model=True
                )
            elif num_chkpt is not None:
                self._checkpoint_manager.keep_last_n_checkpoints(num_chkpt)
        except CheckpointError as ex:
            raise RecipeError(
                f"The previous checkpoints before step {step_nr} cannot be deleted. See the nested exception for details."
            ) from ex

        log.info("Checkpoint at step {} saved.", step_nr)

    def _should_do(
        self,
        after_n_steps: int,
        every_n_steps: int | None,
        after_n_data_epochs: int,
        every_n_data_epochs: int | None,
    ) -> bool:
        if self._state == TrainerState.STOPPING:
            return True

        if self._state == TrainerState.RUNNING:
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

    @override
    def request_stop(self) -> None:
        self._stop_requested = True

        self._state = TrainerState.STOPPING

    @property
    @override
    def step_nr(self) -> int:
        return self._step_nr


class TrainerState(Enum):
    NOT_STARTED = 0
    RUNNING = 1
    STOPPING = 2
    STOPPED = 3


T = TypeVar("T")


class TrainerStateBag(Stateful, Generic[BatchT]):
    _KEYS: Final = [
        "_step_nr",
        "_data_epoch_nr",
        "_lr_scheduler",
        "_loss_scaler",
        "_rng_bag",
        "_metric_bag",
        "_base_wall_time",
        "_has_read_any_data",
    ]

    _trainer: Trainer[BatchT]

    def __init__(self, trainer: Trainer[BatchT]) -> None:
        self._trainer = trainer

    @override
    def state_dict(self) -> dict[str, object]:
        state_dict: dict[str, object] = {}

        def save_stateful(key: str, obj: object) -> None:
            if isinstance(obj, (bool, int, float)):
                state_dict[key] = obj
            elif isinstance(obj, Stateful):
                state_dict[key] = obj.state_dict()
            else:
                raise InternalError(f"`Trainer.{key}` has no state.")

        for key in self._KEYS:
            save_stateful(key, getattr(self._trainer, key))

        return state_dict

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        def load_stateful(key: str) -> None:
            try:
                obj = getattr(self._trainer, key)
            except AttributeError:
                raise InternalError(f"`{key}` is not a `Trainer` attribute.") from None

            try:
                state = state_dict[key]
            except KeyError:
                raise ValueError(f"`state_dict` must contain a key named '{key}'.")

            def type_error(kls: type) -> TypeError:
                raise TypeError(
                    f"`state_dict['{key}']` must be of type `{kls}`, but is of type `{type(state)}` instead."
                )

            if isinstance(obj, (bool, int, float)):
                if type(state) != type(obj):
                    raise type_error(type(obj))

                setattr(self._trainer, key, state)

                return

            if isinstance(obj, Stateful):
                if not isinstance(state, Mapping):
                    raise type_error(Mapping)

                try:
                    obj.load_state_dict(state)
                except (RuntimeError, ValueError, TypeError) as ex:
                    raise ValueError(
                        f"`state_dict['{key}']` is not a valid `{type(obj)}` state. See the nested exception for details."
                    ) from ex

                return

            raise InternalError(f"`Trainer.{key}` has no state.")

        for key in self._KEYS:
            load_stateful(key)
