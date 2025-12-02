# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from contextlib import nullcontext
from enum import Enum
from typing import Any, Final, Generic, Mapping, TypeVar, final

import torch
import torch.distributed
from torch import Tensor
from torch.cuda import OutOfMemoryError
from torch.optim import Optimizer
from torch.profiler import record_function
from typing_extensions import override

from fairseq2.checkpoint import (
    CheckpointDeleteError,
    CheckpointError,
    CheckpointLoadError,
    CheckpointManager,
    CheckpointSaveError,
    CheckpointState,
    HuggingFaceSaveError,
    HuggingFaceSaver,
    Stateful,
)
from fairseq2.data_type import DataType
from fairseq2.datasets import DataReader, DataReadError
from fairseq2.device import CPU, SupportsDeviceTransfer
from fairseq2.error import InternalError, InvalidOperationError
from fairseq2.gang import GangError, Gangs, all_sum, broadcast_flag
from fairseq2.logging import log
from fairseq2.metrics import (
    Mean,
    MetricBag,
    MetricBagError,
    sync_and_compute_metrics,
)
from fairseq2.metrics.recorders import (
    MetricDescriptor,
    MetricRecorder,
    MetricRecordError,
)
from fairseq2.nn.utils.grad import check_grad_norms, normalize_grads
from fairseq2.optim import DynamicLossScaler
from fairseq2.optim.lr_scheduler import LRScheduler, get_effective_lr
from fairseq2.profilers import Profiler
from fairseq2.recipes.metrics import extend_batch_metric_values
from fairseq2.typing import ContextManager
from fairseq2.utils.device_stat import DeviceStatTracker
from fairseq2.utils.gc import GarbageCollector
from fairseq2.utils.progress import ProgressReporter, ProgressTask
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch

# isort: split

from fairseq2.recipes._early_stopper import EarlyStopper
from fairseq2.recipes._error import (
    InconsistentGradNormError,
    MinimumLossScaleReachedError,
    RecipeError,
    UnitError,
)
from fairseq2.recipes._model import Model
from fairseq2.recipes._recipe import Recipe, RecipeStopException
from fairseq2.recipes._validator import Validator

BatchT_contra = TypeVar(
    "BatchT_contra", bound=SupportsDeviceTransfer, contravariant=True
)


class TrainUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Trainer`."""

    def set_step_nr(self, step_nr: int) -> None:
        pass

    def set_data_epoch_nr(self, data_epoch_nr: int) -> None:
        pass

    @abstractmethod
    def __call__(
        self, batch: BatchT_contra, metric_bag: MetricBag
    ) -> tuple[Tensor, int | None]:
        """Process ``batch``.

        :returns:
            - The loss.
            - The number of targets used to compute the loss. If ``None``, the
              model gradients won't be normalized.
        """

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        pass

    @property
    @abstractmethod
    def model(self) -> Model: ...


BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


@final
class Trainer(Recipe):
    """Trains a machine learning model."""

    _state: _TrainerState
    _step_nr: int
    _data_epoch_nr: int
    _unit: TrainUnit[Any]
    _data_reader: DataReader[Any]
    _gangs: Gangs
    _dtype: DataType
    _amp: bool
    _optimizer: Optimizer
    _lr_scheduler: LRScheduler
    _loss_scaler: DynamicLossScaler
    _no_sync_grad_accumulation: bool
    _max_grad_norm: float | None
    _grad_check: bool
    _anomaly_detection: bool
    _rng_bag: RngBag
    _seed: int
    _max_num_steps: int | None
    _max_num_data_epochs: int | None
    _validator: Validator | None
    _validate_at_start: bool
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
    _save_model_only: bool
    _hugging_face_saver: HuggingFaceSaver | None
    _keep_last_n_checkpoints: int | None
    _keep_best_n_checkpoints: int | None
    _keep_checkpoint_every_n_steps: int | None
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
    _first_iter: bool
    _batches: list[Any] | None
    _stop_requested: bool
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
        checkpoint_manager: CheckpointManager,
        metric_recorder: MetricRecorder,
        garbage_collector: GarbageCollector,
        profiler: Profiler,
        device_stat_tracker: DeviceStatTracker,
        wall_watch: Stopwatch,
        progress_reporter: ProgressReporter,
        fp16_loss_scale: tuple[float, float] = (16, 0.0001),
        no_sync_grad_accumulation: bool = False,
        max_grad_norm: float | None = None,
        grad_check: bool = False,
        anomaly_detection: bool = False,
        max_num_steps: int | None = None,
        max_num_data_epochs: int | None = None,
        validator: Validator | None = None,
        validate_at_start: bool = False,
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
        save_model_only: bool = False,
        hugging_face_saver: HuggingFaceSaver | None = None,
        keep_last_n_checkpoints: int | None = None,
        keep_best_n_checkpoints: int | None = None,
        keep_checkpoint_every_n_steps: int | None = None,
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
        :param max_grad_norm:
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
        self._state = _TrainerState.NOT_STARTED

        self._step_nr = 0

        self._data_epoch_nr = 1

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
            grad_accumulation=data_reader.num_accumulate,
            enabled=dtype == torch.float16,
        )

        self._loss_scaler = loss_scaler

        self._no_sync_grad_accumulation = no_sync_grad_accumulation

        self._max_grad_norm = max_grad_norm

        self._grad_check = grad_check

        self._anomaly_detection = anomaly_detection

        self._rng_bag = RngBag.from_device_defaults(CPU, gangs.root.device)

        self._seed = seed

        if max_num_steps is not None:
            if max_num_steps <= 0:
                raise ValueError("`max_num_steps` must be greater than or equal to 1.")

        self._max_num_steps = max_num_steps

        if max_num_data_epochs is not None:
            if max_num_data_epochs <= 0:
                raise ValueError(
                    "`max_num_data_epochs` must be greater than or equal to 1."
                )

        self._max_num_data_epochs = max_num_data_epochs

        self._validator = validator

        self._validate_at_start = validate_at_start

        if validate_every_n_steps is not None:
            if validate_every_n_steps <= 0:
                raise ValueError(
                    "`validate_every_n_steps` must be greater than or equal to 1."
                )

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
                    "`validate_every_n_data_epochs` must be greater than or equal to 1."
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
                    "`checkpoint_every_n_steps` must be greater than or equal to 1."
                )

            if publish_metrics_every_n_steps is not None:
                if checkpoint_every_n_steps % publish_metrics_every_n_steps != 0:
                    raise ValueError(
                        f"`checkpoint_every_n_steps` must be a multiple of `publish_metrics_every_n_steps` ({publish_metrics_every_n_steps}), but is {checkpoint_every_n_steps} instead."
                    )

        self._checkpoint_after_n_steps = checkpoint_after_n_steps
        self._checkpoint_every_n_steps = checkpoint_every_n_steps

        if checkpoint_every_n_data_epochs is not None:
            if checkpoint_every_n_data_epochs <= 0:
                raise ValueError(
                    "`checkpoint_every_n_data_epochs` must be greater than or equal to 1."
                )

            if publish_metrics_every_n_data_epochs is not None:
                if checkpoint_every_n_data_epochs % publish_metrics_every_n_data_epochs != 0:  # fmt: skip
                    raise ValueError(
                        f"`checkpoint_every_n_data_epochs` must be a multiple of `publish_metrics_every_n_data_epochs` ({publish_metrics_every_n_data_epochs}), but is {checkpoint_every_n_data_epochs} instead."
                    )

        self._checkpoint_after_n_data_epochs = checkpoint_after_n_data_epochs
        self._checkpoint_every_n_data_epochs = checkpoint_every_n_data_epochs

        self._save_model_only = save_model_only

        self._hugging_face_saver = hugging_face_saver

        if keep_last_n_checkpoints is not None:
            if keep_last_n_checkpoints <= 0:
                raise ValueError(
                    "`keep_last_n_checkpoints` must be greater than or equal to 1."
                )

        if keep_best_n_checkpoints is not None:
            if keep_best_n_checkpoints <= 0:
                raise ValueError(
                    "`keep_best_n_checkpoints` must be greater than or equal to 1."
                )

            if checkpoint_every_n_steps is not None:
                if validate_every_n_steps is None:
                    raise ValueError(
                        "`validate_every_n_steps` must be specified when `keep_best_n_checkpoints`  and `checkpoint_every_n_steps` are specified."
                    )

                if checkpoint_every_n_steps % validate_every_n_steps != 0:
                    raise ValueError(
                        f"`checkpoint_every_n_steps` must be a multiple of `validate_every_n_steps` ({validate_every_n_steps}) when `keep_best_n_checkpoints` is specified, but is {checkpoint_every_n_steps} instead."
                    )

        if keep_checkpoint_every_n_steps is not None:
            if keep_checkpoint_every_n_steps <= 0:
                raise ValueError(
                    "`keep_checkpoint_every_n_steps` must be greater than or equal to 1."
                )

            if checkpoint_every_n_steps is not None:
                if keep_checkpoint_every_n_steps % checkpoint_every_n_steps != 0:
                    raise ValueError(
                        f"`keep_checkpoint_every_n_steps` must be a multiple of `checkpoint_every_n_steps` ({checkpoint_every_n_steps}), but is {keep_checkpoint_every_n_steps} instead."
                    )

        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._keep_best_n_checkpoints = keep_best_n_checkpoints

        self._keep_checkpoint_every_n_steps = keep_checkpoint_every_n_steps

        self._metric_bag = MetricBag(device=gangs.root.device)

        self._metric_recorder = metric_recorder

        if publish_metrics_every_n_steps == 0:
            raise ValueError(
                "`publish_metrics_every_n_steps` must be greater than or equal to 1."
            )

        self._publish_metrics_after_n_steps = publish_metrics_after_n_steps
        self._publish_metrics_every_n_steps = publish_metrics_every_n_steps

        if publish_metrics_every_n_data_epochs == 0:
            raise ValueError(
                "`publish_metrics_every_n_data_epochs` must be greater than or equal to 1."
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

        self._first_iter = True

        self._batches = None

        self._stop_requested = False

        self._num_batches_read = 0

        self._last_lr = 0.0

        self._multi_loss_norm = getattr(self._unit, "multi_loss_norm", False)

    @override
    def run(self) -> None:
        if self._state != _TrainerState.NOT_STARTED:
            raise InvalidOperationError("The trainer cannot be run more than once.")

        gangs = self._gangs

        with self._progress_reporter:
            with self._rng_bag.temporary_manual_seed(self._seed + gangs.root.rank):
                self._state = self._maybe_restore_state()

                with self._profiler, self._garbage_collector:
                    self._do_run()

        if self._state != _TrainerState.STOPPED:
            raise InternalError(
                f"`_state` must be `STOPPED`, but is `{self._state}` instead."
            )

        gangs.close()

        if self._stop_requested:
            raise RecipeStopException()

    def _maybe_restore_state(self) -> _TrainerState:
        try:
            step_nr = self._checkpoint_manager.maybe_get_last_step_number(
                exclude_model_only=True
            )
        except CheckpointError as ex:
            raise RecipeError(
                "The checkpoints cannot accessed. See the nested exception for details."
            ) from ex

        if step_nr is None:
            if self._validate_at_start:
                return _TrainerState.PRE_VALIDATION

            return _TrainerState.DATA_LOAD

        log.info("Restoring training from the last checkpoint at step {}.", step_nr)

        try:
            log.info("Restoring the trainer state.")

            trainer_state_bag = _TrainerStateBag(self)

            self._checkpoint_manager.load_trainer_state(step_nr, trainer_state_bag)

            log.info("Trainer state restored.")

            log.info("Restoring the model state.")

            self._checkpoint_manager.load_model_state(step_nr, self._unit.model)

            log.info("Model state restored.")

            log.info("Restoring the optimizer state.")

            self._checkpoint_manager.load_optimizer_state(step_nr, self._optimizer)  # type: ignore[arg-type]

            log.info("Optimizer state restored.")

            log.info("Restoring the data reader state.")

            self._checkpoint_manager.load_data_reader_state(step_nr, self._data_reader)

            log.info("Data reader state restored.")
        except CheckpointLoadError as ex:
            raise RecipeError(
                f"The last checkpoint at step {ex.step_nr} cannot be loaded. See the nested exception for details."
            ) from ex

        self._reset_non_total_metrics()

        try:
            self._gangs.root.barrier()
        except GangError as ex:
            raise RecipeError(
                "The collective barrier after the checkpoint load operation has failed. See the nested exception for details."
            ) from ex

        log.info("Training restored. Resuming from step {}.", step_nr)

        if self._max_num_steps is not None:
            if self._step_nr >= self._max_num_steps:
                return _TrainerState.STOPPED

        if self._max_num_data_epochs is not None:
            if self._data_epoch_nr >= self._max_num_data_epochs:
                return _TrainerState.STOPPED

        return _TrainerState.DATA_LOAD

    def _do_run(self) -> None:
        self._unit.model.module.train()

        progress_task = self._progress_reporter.create_task(
            "train", total=self._max_num_steps, completed=self._step_nr
        )

        try:
            self._unit.set_data_epoch_nr(self._data_epoch_nr)
        except UnitError as ex:
            raise RecipeError(
                "The train unit has failed. See the nested exception for details."
            ) from ex

        self._device_stat_tracker.reset()

        with progress_task, self._lapse_watch:
            while self._state != _TrainerState.STOPPED:
                match self._state:
                    case _TrainerState.PRE_VALIDATION:
                        self._state = self._pre_validate()

                    case _TrainerState.DATA_LOAD:
                        self._state = self._read_next_batches()

                    case _TrainerState.END_OF_DATA_EPOCH:
                        self._state = self._handle_end_of_data_epoch()

                    case _TrainerState.STEP:
                        self._state = self._run_step(progress_task)

                    case _TrainerState.POST_STEP:
                        self._state = self._run_post_step()

                    case _TrainerState.GRAD_OVERFLOW:
                        self._state = self._repeat_step(progress_task)

                    case _TrainerState.END_OF_TRAINING:
                        log.info("End of training reached at step {}.", self._step_nr)

                        self._state = self._stop()

                    case _TrainerState.END_OF_DATA:
                        log.info("End of data reached.")

                        self._state = self._stop()

                    case _TrainerState.EARLY_STOP:
                        self._state = self._early_stop()

                    case _TrainerState.STOP_REQUESTED:
                        log.info("Stopping training at step {}.", self._step_nr)

                        self._state = self._stop()

    def _pre_validate(self) -> _TrainerState:
        if self._validate is not None:
            self._validate()

        return _TrainerState.DATA_LOAD

    def _read_next_batches(self) -> _TrainerState:
        with self._data_watch:
            try:
                self._batches = next(self._data_reader)
            except DataReadError as ex:
                raise RecipeError(
                    "The train data read operation has failed. See the nested exception for details."
                ) from ex
            except StopIteration:
                self._batches = None

            if self._batches is not None:
                return _TrainerState.STEP

            self._data_reader.reset()

        if self._step_nr == 0:
            log.info("Dataset is empty or too small to train. Stopping training.")

            return _TrainerState.STOPPED
        else:
            return _TrainerState.END_OF_DATA_EPOCH

    def _handle_end_of_data_epoch(self) -> _TrainerState:
        log.info("End of epoch {} reached after step {}.", self._data_epoch_nr, self._step_nr)  # fmt: skip

        if self._max_num_data_epochs is not None:
            if self._data_epoch_nr >= self._max_num_data_epochs:
                return _TrainerState.END_OF_DATA

        state = self._run_post_step()

        self._data_epoch_nr += 1

        try:
            self._unit.set_data_epoch_nr(self._data_epoch_nr)
        except UnitError as ex:
            raise RecipeError(
                "The train unit has failed. See the nested exception for details."
            ) from ex

        return state

    def _run_step(self, progress_task: ProgressTask) -> _TrainerState:
        try:
            self._checkpoint_manager.maybe_complete_async_checkpoint()
        except CheckpointSaveError as ex:
            raise RecipeError(
                f"The checkpoint of step {ex.step_nr} cannot be saved. See the nested exception for details."
            ) from ex

        detect_anomaly = torch.autograd.set_detect_anomaly(  # type: ignore[attr-defined]
            self._anomaly_detection, check_nan=True
        )

        self._step_nr += 1

        progress_task.step()

        with detect_anomaly:
            with record_function(f"step_{self._step_nr}"):
                state = self._do_run_step(progress_task)

        self._profiler.step()

        self._garbage_collector.step()

        if self._first_iter:
            # Emptying the CUDA memory allocator cache after the first iteration
            # can reduce fragmentation and avoid OOM.
            if self._gangs.root.device.type == "cuda":
                torch.cuda.empty_cache()

            self._first_iter = False

        return state

    def _do_run_step(self, progress_task: ProgressTask) -> _TrainerState:
        step_nr = self._step_nr

        log.debug("Running step {}.", step_nr)

        batches = self._batches
        if batches is None:
            raise InternalError("`_batches` is `None`.")

        self._batches = None

        if self._loss_scaler.is_enabled:
            self._metric_bag.begin_updates()

        gangs = self._gangs

        num_targets = 0

        with self._compute_watch:
            with record_function(f"step_{step_nr}_setup"):
                try:
                    self._unit.set_step_nr(step_nr)
                except UnitError as ex:
                    raise RecipeError(
                        "The train unit has failed. See the nested exception for details."
                    ) from ex

            batches.reverse()

            num_batches = len(batches)

            for batch_nr in range(num_batches):
                batch = batches.pop()

                try:
                    batch.to(gangs.root.device, non_blocking=True)

                    with self._maybe_no_sync(batch_nr, num_batches):
                        if self._multi_loss_norm:
                            assert (
                                num_batches == 1
                            ), "microbatching is not supported for multiple loss norm yet"
                            with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                                loss_target_count_dict = self._compute_loss(batch)

                            all_losses = []
                            for name, (
                                curr_loss,
                                target_count,
                            ) in loss_target_count_dict.items():
                                if target_count is None:
                                    target_sum = num_batches
                                else:
                                    # we do the all sum here as compared to in
                                    # grad scaling to apply different norm to
                                    # different loss components.
                                    # TODO(lidli): double check if we need to consider the factor of world size like in grad scale.
                                    target_sum = all_sum(gangs.dp, target_count)
                                curr_loss = curr_loss * gangs.dp.size / target_sum
                                all_losses.append(curr_loss)
                                log.info(f"{name}_loss={curr_loss}, {target_sum=}")
                                self._metric_bag.get(Mean, f"{name}_after_norm").update(
                                    curr_loss / batch.batch_size,
                                    weight=batch.batch_size,
                                )
                                log.info(f"{all_losses=}")
                            loss = sum(all_losses)
                        else:
                            with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                                loss, num_batch_targets = self._compute_loss(batch)

                            # If the unit does not return the number of logit targets
                            # of this batch, we assume that the loss is the mean loss
                            # and that each batch in this step has the same number of
                            # logit targets. In this case, we don't need to normalize
                            # the gradients at the end of the step, but we still have
                            # to take gradient accumulation into account.
                            if num_batch_targets is None:
                                loss = loss / num_batches
                            else:
                                num_targets += num_batch_targets

                        with record_function(f"step_{step_nr}_{batch_nr}_backward"):
                            self._loss_scaler.backward(loss)

                        del loss
                except OutOfMemoryError:
                    log.error("CUDA out of memory. Note that CUDA operations are async. Dumping the likely input batch. Use `CUDA_LAUNCH_BLOCKING=1` for debugging:\n{}", batch)  # fmt: skip

                    raise

            # This function gathers the total number of logit targets across all
            # processes and divides the gradients by it. This is needed when the
            # batches have varying sizes and we cannot normalize the loss before
            # the backward pass.
            if num_targets > 0:
                normalize_grads(self._unit.model.module, gangs.dp, num_targets)

            self._loss_scaler.unscale_grads_()

            # Clip the gradients.
            with record_function(f"step_{step_nr}_grad_norm"):
                grad_norm = self._unit.model.clip_grad_norm(self._max_grad_norm)

                if self._grad_check:
                    if not check_grad_norms(grad_norm, gangs.dp, step_nr):
                        raise InconsistentGradNormError(step_nr)

            # Update the parameters.
            with record_function(f"step_{step_nr}_optimizer"):
                _, scale_result = self._loss_scaler.run_optimizer_step(step_nr)

            self._optimizer.zero_grad(set_to_none=True)

        if scale_result.overflow:
            if scale_result.min_reached:
                raise MinimumLossScaleReachedError(step_nr)

            self._metric_bag.rollback_updates()

            return _TrainerState.GRAD_OVERFLOW

        self._last_lr = get_effective_lr(self._lr_scheduler)

        self._lr_scheduler.step()

        if self._loss_scaler.is_enabled:
            self._metric_bag.commit_updates()

        self._metric_bag.get(Mean, "grad_norm").update(grad_norm)

        self._num_batches_read += 1

        return _TrainerState.POST_STEP

    def _maybe_no_sync(self, batch_nr: int, num_batches: int) -> ContextManager:
        if self._no_sync_grad_accumulation:
            if batch_nr < num_batches - 1:
                return self._unit.model.no_sync()

        return nullcontext()

    def _compute_loss(self, batch: Any) -> tuple[Tensor, int | None]:
        with self._maybe_autocast():
            try:
                return self._unit(batch, self._metric_bag)
            except UnitError as ex:
                raise RecipeError(
                    "The train unit has failed. See the nested exception for details."
                ) from ex

    def _maybe_autocast(self) -> ContextManager:
        if self._dtype == torch.float32 or not self._amp:
            return nullcontext()

        device_type = self._gangs.root.device.type

        return torch.autocast(device_type=device_type, dtype=self._dtype)

    def _repeat_step(self, progress_task: ProgressTask) -> _TrainerState:
        if self._stop_requested:
            return _TrainerState.STOP_REQUESTED

        log.debug("Repeating step {}.", self._step_nr)

        self._step_nr -= 1

        progress_task.step(-1)

        return _TrainerState.DATA_LOAD

    def _run_post_step(self) -> _TrainerState:
        if self._max_num_steps is not None:
            if self._step_nr >= self._max_num_steps:
                return _TrainerState.END_OF_TRAINING
        elif self._stop_requested:
            return _TrainerState.STOP_REQUESTED

        self._maybe_checkpoint(blocking=False)

        self._maybe_publish_metrics()

        score = self._maybe_validate()

        if score is not None:
            should_stop = self._maybe_request_early_stop(score)
        else:
            should_stop = False

        if should_stop:
            return _TrainerState.EARLY_STOP
        else:
            return _TrainerState.DATA_LOAD

    def _stop(self) -> _TrainerState:
        self._maybe_publish_metrics()

        should_validate = self._should_validate()
        if should_validate:
            self._maybe_checkpoint(blocking=False)

            self._validate()

            self._maybe_wait_checkpoint()
        else:
            self._maybe_checkpoint(blocking=True)

        return _TrainerState.STOPPED

    def _early_stop(self) -> _TrainerState:
        log.info("Early stop requested. Stopping training at step {}.", self._step_nr)  # fmt: skip

        self._stop_requested = True

        should_checkpoint = self._should_checkpoint()
        if should_checkpoint:
            self._checkpoint(blocking=True)
        else:
            self._maybe_wait_checkpoint()

        return _TrainerState.STOPPED

    def _maybe_checkpoint(self, blocking: bool) -> None:
        should_checkpoint = self._should_checkpoint()
        if should_checkpoint:
            self._checkpoint(blocking)

    def _should_checkpoint(self) -> bool:
        return self._should_do(
            self._checkpoint_after_n_steps,
            self._checkpoint_every_n_steps,
            self._checkpoint_after_n_data_epochs,
            self._checkpoint_every_n_data_epochs,
        )

    def _checkpoint(self, blocking: bool) -> None:
        step_nr = self._step_nr

        log.info("Preparing checkpoint at step {}.", step_nr)

        self._maybe_wait_checkpoint()

        def log_ready(step_nr: int, state: CheckpointState) -> None:
            if blocking:
                log.info("Checkpoint prepared. Saving.")
            else:
                log.info("Checkpoint prepared. Saving asynchronously.")

        try:
            if self._save_model_only:
                self._checkpoint_manager.save_model_only(
                    step_nr,
                    self._unit.model,
                    state_processor=log_ready,
                    callback=self._complete_checkpoint,
                    blocking=blocking,
                )
            else:
                shim_trainer = _TrainerStateBag(self)

                self._checkpoint_manager.save_checkpoint(
                    step_nr,
                    shim_trainer,
                    self._unit.model,
                    self._optimizer,  # type: ignore[arg-type]
                    self._data_reader,
                    state_processor=log_ready,
                    callback=self._complete_checkpoint,
                    blocking=blocking,
                )
        except CheckpointSaveError as ex:
            raise RecipeError(
                f"The checkpoint of step {ex.step_nr} cannot be saved. See the nested exception for details."
            ) from ex

    def _complete_checkpoint(self, step_nr: int, blocking: bool) -> None:
        log.info("Checkpoint at step {} saved.", step_nr)

        gangs = self._gangs

        hg_saver = self._hugging_face_saver

        if hg_saver is not None:
            if gangs.root.rank == 0:
                if hg_saver.is_saving:
                    log.info("Waiting for the current Hugging Face model save operation to complete before continuing.")  # fmt: skip

                try:
                    hg_saver.complete_pending()
                except HuggingFaceSaveError as ex:
                    raise RecipeError(
                        f"The Hugging Face model of step {ex.step_nr} cannot be saved. See the nested exception for details."  # fmt: skip
                    ) from ex

            try:
                gangs.root.barrier()
            except GangError as ex:
                raise RecipeError(
                    "The collective barrier after the Hugging Face wait operation has failed. See the nested exception for details."
                ) from ex

        self._delete_stale_checkpoints()

        if hg_saver is not None:
            if gangs.root.rank == 0:
                if blocking:
                    log.info("Saving Hugging Face model of step {}.", step_nr)  # fmt: skip
                else:
                    log.info("Asynchronously saving Hugging Face model of step {}.", step_nr)  # fmt: skip

                def save_callback(step_nr: int) -> None:
                    log.info("Hugging Face model of step {} saved.", step_nr)

                hg_saver.save(step_nr, callback=save_callback, blocking=blocking)

    def _maybe_publish_metrics(self) -> None:
        should_publish = self._should_publish_metrics()
        if should_publish:
            self._publish_metrics()

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
                values = sync_and_compute_metrics(self._metric_bag, gangs.dp)
            else:
                values = None
        except MetricBagError as ex:
            raise RecipeError(
                "The train metric values cannot be synced across processes. See the nested exception for details."
            ) from ex

        if gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            self._unit.process_metric_values(values)

            values["lr"] = self._last_lr

            values["data_epoch"] = self._data_epoch_nr

            device_stats = self._device_stat_tracker.get_stats()

            values.update(device_stats)

            data_time = self._data_watch.get_elapsed_time()

            compute_time = self._compute_watch.get_elapsed_time()

            extend_batch_metric_values(
                values, self._num_batches_read, data_time + compute_time
            )

            values["data_time"] = data_time

            values["compute_time"] = compute_time

            values["lapse_time"] = self._lapse_watch.get_elapsed_time()

            wall_time = self._wall_watch.get_elapsed_time()

            values["total_time"] = self._base_wall_time + wall_time

            values["wall_time"] = wall_time

            try:
                self._metric_recorder.record_metric_values(
                    "train", values, self._step_nr
                )
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

        self._reset_lapse_state()

    def _reset_lapse_state(self) -> None:
        self._reset_non_total_metrics()

        self._data_watch.reset()

        self._compute_watch.reset()

        self._lapse_watch.reset()

        self._device_stat_tracker.reset()

        self._num_batches_read = 0

    def _reset_non_total_metrics(self) -> None:
        for name, metric in self._metric_bag.metrics.items():
            if not name.startswith("total_"):
                metric.reset()

    def _maybe_validate(self) -> float | None:
        should_validate = self._should_validate()
        if should_validate:
            self._lapse_watch.stop()

            score = self._validate()

            self._lapse_watch.start()
        else:
            score = None

        return score

    def _should_validate(self) -> bool:
        if self._validator is None:
            return False

        return self._should_do(
            self._validate_after_n_steps,
            self._validate_every_n_steps,
            self._validate_after_n_data_epochs,
            self._validate_every_n_data_epochs,
        )

    def _validate(self) -> float | None:
        if self._validator is None:
            raise InternalError("`_validator` is `None`.")

        if self._step_nr == 0:
            log.info("Starting pre-validation before training.")
        else:
            log.info("Starting validation after step {}.", self._step_nr)

        with self._unit.model.summon_full_parameters():
            score = self._validator.run(self._step_nr)

        if score is not None:
            if self._should_checkpoint():
                self._save_score(score)

        self._validator.reset()

        self._unit.model.module.train()

        # Try to avoid CUDA memory fragmentation after validation.
        if self._gangs.root.device.type == "cuda":
            torch.cuda.empty_cache()

        log.info("Validation finished.")

        return score

    def _save_score(self, score: float) -> None:
        try:
            self._checkpoint_manager.save_score(self._step_nr, score)
        except CheckpointSaveError as ex:
            raise RecipeError(
                f"The score of step {self._step_nr} cannot be saved. See the nested exception for details."
            ) from ex

        if self._keep_best_n_checkpoints is not None:
            self._delete_stale_checkpoints()

    def _maybe_request_early_stop(self, score: float) -> bool:
        if self._early_stopper is None:
            return False

        gangs = self._gangs

        if gangs.root.rank == 0:
            should_stop = self._early_stopper.should_stop(self._step_nr, score)
        else:
            should_stop = False

        return broadcast_flag(gangs.root, should_stop)

    def _maybe_wait_checkpoint(self) -> None:
        if not self._checkpoint_manager.is_saving:
            return

        log.info("Waiting for the current checkpoint save operation to complete before continuing.")  # fmt: skip

        try:
            self._checkpoint_manager.maybe_complete_async_checkpoint(blocking=True)
        except CheckpointSaveError as ex:
            raise RecipeError(
                f"The checkpoint of step {ex.step_nr} cannot be saved. See the nested exception for details."
            ) from ex

    def _delete_stale_checkpoints(self) -> None:
        try:
            stale_step_nrs = self._checkpoint_manager.get_stale_step_numbers(
                self._keep_last_n_checkpoints,
                self._keep_best_n_checkpoints,
                self._keep_checkpoint_every_n_steps,
            )
        except CheckpointError as ex:
            raise RecipeError(
                "The stale checkpoints cannot be determined. See the nested exception for details."
            ) from ex

        if not stale_step_nrs:
            return

        log.info("Deleting stale checkpoints.")

        for step_nr in stale_step_nrs:
            try:
                self._checkpoint_manager.delete_checkpoint(step_nr)
            except CheckpointDeleteError as ex:
                raise RecipeError(
                    f"The checkpoint of step {ex.step_nr} cannot be deleted. See the nested exception for details."
                ) from ex

        log.info("Stale checkpoints deleted.")

    def _should_do(
        self,
        after_n_steps: int,
        every_n_steps: int | None,
        after_n_data_epochs: int,
        every_n_data_epochs: int | None,
    ) -> bool:
        if self._state == _TrainerState.PRE_VALIDATION:
            return False

        def should_do_at_step() -> bool:
            if every_n_steps is not None:
                if self._step_nr > after_n_steps:
                    if self._step_nr % every_n_steps == 0:
                        return True

            return False

        if self._state == _TrainerState.POST_STEP:
            return should_do_at_step()

        if self._state == _TrainerState.END_OF_TRAINING:
            return True

        if self._state == _TrainerState.END_OF_DATA:
            already_done = should_do_at_step()

            # If we have already returned true for this step before reaching the
            # end of data, we should return false this time.
            return not already_done

        if self._state == _TrainerState.END_OF_DATA_EPOCH:
            if every_n_data_epochs is not None:
                if self._data_epoch_nr > after_n_data_epochs:
                    if self._data_epoch_nr % every_n_data_epochs == 0:
                        already_done = should_do_at_step()

                        # Same condition as above for `END_OF_DATA`.
                        return not already_done

            return False

        if self._state == _TrainerState.STOP_REQUESTED:
            return True

        if self._state == _TrainerState.EARLY_STOP:
            already_done = should_do_at_step()

            # If we have already returned true for this step before an early
            # stop, we should return false this time.
            return not already_done

        raise InternalError(f"`_state` is `{self._state}`")

    @override
    def request_stop(self) -> None:
        self._stop_requested = True

    @override
    def close(self) -> None:
        self._checkpoint_manager.close()

        self._metric_recorder.close()

    @property
    @override
    def step_nr(self) -> int:
        return self._step_nr


class _TrainerState(Enum):
    NOT_STARTED = 0
    PRE_VALIDATION = 1
    DATA_LOAD = 2
    STEP = 3
    POST_STEP = 4
    END_OF_DATA_EPOCH = 5
    END_OF_TRAINING = 6
    END_OF_DATA = 7
    GRAD_OVERFLOW = 8
    EARLY_STOP = 9
    STOP_REQUESTED = 10
    STOPPED = 11


T = TypeVar("T")


class _TrainerStateBag(Stateful):
    _ATTR_NAMES: Final = [
        "_step_nr",
        "_data_epoch_nr",
        "_lr_scheduler",
        "_loss_scaler",
        "_rng_bag",
        "_metric_bag",
    ]

    _trainer: Trainer

    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer

    @override
    def state_dict(self) -> dict[str, object]:
        state_dict: dict[str, object] = {}

        def save_stateful(name: str, value: object) -> None:
            if isinstance(value, Stateful):
                state_dict[name] = value.state_dict()
            else:
                state_dict[name] = value

        for name in self._ATTR_NAMES:
            value = getattr(self._trainer, name)

            save_stateful(name, value)

        wall_time = self._trainer._wall_watch.get_elapsed_time()

        state_dict["_base_wall_time"] = self._trainer._base_wall_time + wall_time

        return state_dict

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        def load_stateful(name: str) -> None:
            value = getattr(self._trainer, name)

            try:
                state = state_dict[name]
            except KeyError:
                raise ValueError(f"`state_dict` must contain a key named '{name}'.")

            def type_error(kls: type) -> TypeError:
                raise TypeError(
                    f"`state_dict['{name}']` must be of type `{kls}`, but is of type `{type(state)}` instead."
                )

            kls = type(value)

            if isinstance(value, Stateful):
                if not isinstance(state, Mapping):
                    raise type_error(Mapping)

                try:
                    value.load_state_dict(state)
                except (RuntimeError, ValueError, TypeError) as ex:
                    raise ValueError(
                        f"`state_dict['{name}']` is not a valid `{kls}` state. See the nested exception for details."
                    ) from ex
            else:
                if type(state) != kls:
                    raise type_error(kls)

                setattr(self._trainer, name, state)

        for name in self._ATTR_NAMES:
            load_stateful(name)

        load_stateful("_base_wall_time")
