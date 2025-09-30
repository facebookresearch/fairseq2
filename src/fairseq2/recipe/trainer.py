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
from typing import Any, Generic, Literal, TypeVar, final

import torch
import torch.distributed
from torch import Tensor
from torch.cuda import OutOfMemoryError
from torch.optim import Optimizer
from torch.profiler import record_function
from typing_extensions import override

from fairseq2.checkpoint import (
    NOOP_CHECKPOINT_HG_EXPORTER,
    CheckpointHGExporter,
    CheckpointManager,
)
from fairseq2.data_type import DataType
from fairseq2.datasets import DataReader
from fairseq2.device import CPU, SupportsDeviceTransfer
from fairseq2.early_stopper import EarlyStopper
from fairseq2.error import InternalError, InvalidOperationError, StateDictError
from fairseq2.gang import GangError, Gangs, broadcast_flag, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.metrics import Mean, MetricBag, sync_and_compute_metrics
from fairseq2.metrics.common import extend_batch_metric_values
from fairseq2.metrics.recorders import MetricRecorder
from fairseq2.nn.utils.grad import check_grad_norms, normalize_grads
from fairseq2.optim.fp16_loss_scaler import (
    NOOP_FP16_LOSS_SCALER,
    Float16LossScaler,
    Float16LossScaleResult,
)
from fairseq2.optim.lr_schedulers import LRScheduler
from fairseq2.profilers import Profiler
from fairseq2.recipe.error import MinimumLossScaleReachedError
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.task import Task, TaskStopException
from fairseq2.recipe.validator import NOOP_VALIDATOR, Validator
from fairseq2.typing import ContextManager, Stateful
from fairseq2.utils.device_stat import DeviceStatTracker
from fairseq2.utils.gc import GarbageCollector
from fairseq2.utils.progress import ProgressReporter, ProgressTask
from fairseq2.utils.rng import RngBag
from fairseq2.utils.stopwatch import Stopwatch

BatchT_contra = TypeVar(
    "BatchT_contra", bound=SupportsDeviceTransfer, contravariant=True
)


class TrainUnit(ABC, Generic[BatchT_contra]):
    """Represents a unit to be used with :class:`Trainer`."""

    @abstractmethod
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None: ...

    def set_step_nr(self, step_nr: int) -> None:
        pass

    def set_data_epoch_nr(self, data_epoch_nr: int) -> None:
        pass

    @abstractmethod
    def process_batch(
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
    def model(self) -> RecipeModel: ...


BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


@final
class Trainer(Task):
    """Trains a machine learning model."""

    def __init__(
        self,
        *,
        unit: TrainUnit[BatchT],
        data_reader: DataReader[BatchT],
        gangs: Gangs,
        amp: bool,
        amp_dtype: DataType,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        fp16_loss_scaler: Float16LossScaler,
        validator: Validator,
        checkpoint_manager: CheckpointManager,
        checkpoint_hg_exporter: CheckpointHGExporter,
        metric_recorder: MetricRecorder,
        garbage_collector: GarbageCollector,
        profiler: Profiler,
        device_stat_tracker: DeviceStatTracker,
        wall_watch: Stopwatch,
        progress_reporter: ProgressReporter,
        seed: int,
        no_sync_grad_accumulation: bool = False,
        max_grad_norm: float | None = None,
        grad_check: bool = False,
        anomaly_detection: bool = False,
        max_num_steps: int | None = None,
        max_num_data_epochs: int | None = None,
        validate_at_start: bool = False,
        validate_after_n_steps: int = 0,
        validate_every_n_steps: int | None = None,
        validate_after_n_data_epochs: int = 0,
        validate_every_n_data_epochs: int | None = None,
        early_stopper: EarlyStopper | None = None,
        checkpoint_after_n_steps: int = 0,
        checkpoint_every_n_steps: int | None = None,
        checkpoint_after_n_data_epochs: int = 0,
        checkpoint_every_n_data_epochs: int | None = None,
        save_model_only: bool | Literal["all", "all_but_last"] = False,
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
        :param lr_scheduler:
            The learning rate scheduler.
        :param amp:
            If ``True``, enables ``torch.amp``.
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
        if max_num_steps is not None:
            if max_num_steps <= 0:
                raise ValueError("`max_num_steps` must be greater than or equal to 1.")

        if max_num_data_epochs is not None:
            if max_num_data_epochs <= 0:
                raise ValueError(
                    "`max_num_data_epochs` must be greater than or equal to 1."
                )

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

        if publish_metrics_every_n_steps == 0:
            raise ValueError(
                "`publish_metrics_every_n_steps` must be greater than or equal to 1."
            )

        if publish_metrics_every_n_data_epochs == 0:
            raise ValueError(
                "`publish_metrics_every_n_data_epochs` must be greater than or equal to 1."
            )

        last_lrs = [0.0] * len(optimizer.param_groups)

        self._state = _TrainerState.NOT_STARTED
        self._step_nr = 0
        self._data_epoch_nr = 1
        self._unit = unit
        self._data_reader = data_reader
        self._gangs = gangs
        self._amp = amp
        self._amp_dtype = amp_dtype
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._fp16_loss_scaler = fp16_loss_scaler
        self._no_sync_grad_accumulation = no_sync_grad_accumulation
        self._max_grad_norm = max_grad_norm
        self._grad_check = grad_check
        self._anomaly_detection = anomaly_detection
        self._rng_bag = RngBag.from_device_defaults(CPU, gangs.root.device)
        self._max_num_steps = max_num_steps
        self._max_num_data_epochs = max_num_data_epochs
        self._validator = validator
        self._validate_at_start = validate_at_start
        self._validate_after_n_steps = validate_after_n_steps
        self._validate_every_n_steps = validate_every_n_steps
        self._validate_after_n_data_epochs = validate_after_n_data_epochs
        self._validate_every_n_data_epochs = validate_every_n_data_epochs
        self._early_stopper = early_stopper
        self._checkpoint_manager = checkpoint_manager
        self._checkpoint_after_n_steps = checkpoint_after_n_steps
        self._checkpoint_every_n_steps = checkpoint_every_n_steps
        self._checkpoint_after_n_data_epochs = checkpoint_after_n_data_epochs
        self._checkpoint_every_n_data_epochs = checkpoint_every_n_data_epochs
        self._save_model_only = save_model_only
        self._checkpoint_hg_exporter = checkpoint_hg_exporter
        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._keep_best_n_checkpoints = keep_best_n_checkpoints
        self._keep_checkpoint_every_n_steps = keep_checkpoint_every_n_steps
        self._metric_bag = MetricBag(device=gangs.root.device)
        self._metric_recorder = metric_recorder
        self._publish_metrics_after_n_steps = publish_metrics_after_n_steps
        self._publish_metrics_every_n_steps = publish_metrics_every_n_steps
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
        self._seed = seed
        self._first_iter = True
        self._batches: list[Any] | None = None
        self._stop_requested = False
        self._num_batches_read = 0
        self._last_lrs = last_lrs

        self._metric_bag.add("grad_norm", Mean())

        unit.prepare_metric_bag(self._metric_bag)

    @override
    def run(self) -> None:
        if self._state != _TrainerState.NOT_STARTED:
            raise InvalidOperationError("Trainer cannot be run more than once.")

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
            raise TaskStopException()

    def _maybe_restore_state(self) -> _TrainerState:
        step_nr = self._checkpoint_manager.maybe_get_last_step_number(
            exclude_model_only=True
        )

        if step_nr is None:
            if self._validate_at_start:
                return _TrainerState.PRE_VALIDATION

            return _TrainerState.DATA_LOAD

        log.info("Restoring training from the last checkpoint at step {}.", step_nr)

        log.info("Restoring trainer state.")

        trainer = _TrainerCheckpointProxy(self)

        self._checkpoint_manager.load_trainer_state(step_nr, trainer)

        log.info("Trainer state restored.")

        log.info("Restoring model state.")

        self._checkpoint_manager.load_model_state(step_nr, self._unit.model)

        log.info("Model state restored.")

        log.info("Restoring optimizer state.")

        self._checkpoint_manager.load_optimizer_state(step_nr, self._optimizer)

        log.info("Optimizer state restored.")

        log.info("Restoring data reader state.")

        self._checkpoint_manager.load_data_reader_state(step_nr, self._data_reader)

        log.info("Data reader state restored.")

        self._reset_non_total_metrics()

        self._gangs.root.barrier()

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

        self._unit.set_data_epoch_nr(self._data_epoch_nr)

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

        self._unit.set_data_epoch_nr(self._data_epoch_nr)

        return state

    def _run_step(self, progress_task: ProgressTask) -> _TrainerState:
        self._checkpoint_manager.maybe_complete_save_operation()

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

        if self._fp16_loss_scaler is not NOOP_FP16_LOSS_SCALER:
            self._metric_bag.begin_updates()

        gangs = self._gangs

        num_targets = 0

        with self._compute_watch:
            with record_function(f"step_{step_nr}_setup"):
                self._unit.set_step_nr(step_nr)

            batches.reverse()

            num_batches = len(batches)

            for batch_nr in range(num_batches):
                batch = batches.pop()

                try:
                    batch.to(gangs.root.device, non_blocking=True)

                    with self._maybe_no_sync(batch_nr, num_batches):
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
                            self._fp16_loss_scaler.backward(loss)

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

            self._fp16_loss_scaler.unscale_grads_(self._optimizer)

            # Clip the gradients.
            with record_function(f"step_{step_nr}_grad_norm"):
                grad_norm = self._unit.model.clip_grad_norm(self._max_grad_norm)

                if self._grad_check:
                    check_grad_norms(grad_norm, gangs.dp, step_nr)

            # Update the parameters.
            with record_function(f"step_{step_nr}_optimizer"):
                fp16_scale_result = self._fp16_loss_scaler.run_optimizer_step(
                    self._optimizer
                )

            self._optimizer.zero_grad(set_to_none=True)

        self._inspect_fp16_scale_result(fp16_scale_result)

        if fp16_scale_result.overflowed:
            self._metric_bag.rollback_updates()

            return _TrainerState.GRAD_OVERFLOW

        self._last_lrs = self._lr_scheduler.get_last_lr()

        self._lr_scheduler.step()

        if self._fp16_loss_scaler is not NOOP_FP16_LOSS_SCALER:
            self._metric_bag.commit_updates()

        self._metric_bag.get("grad_norm", Mean).update(grad_norm)

        self._num_batches_read += 1

        return _TrainerState.POST_STEP

    def _maybe_no_sync(self, batch_nr: int, num_batches: int) -> ContextManager[None]:
        if self._no_sync_grad_accumulation:
            if batch_nr < num_batches - 1:
                return self._unit.model.no_sync()

        return nullcontext()

    def _compute_loss(self, batch: Any) -> tuple[Tensor, int | None]:
        with self._maybe_autocast():
            return self._unit.process_batch(batch, self._metric_bag)

    def _maybe_autocast(self) -> ContextManager[None]:
        if not self._amp or self._amp_dtype == torch.float32:
            return nullcontext()

        device_type = self._gangs.root.device.type

        return torch.autocast(device_type=device_type, dtype=self._amp_dtype)

    def _inspect_fp16_scale_result(self, result: Float16LossScaleResult) -> None:
        if result.exploded:
            log.error("Overflow detected at step {}, ignoring gradient, loss scale is already at minimum ({:g}). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping, or increasing the batch size.", self._step_nr, result.new_scale)  # fmt: skip

            raise MinimumLossScaleReachedError(self._step_nr)

        if result.scaled:
            log.info("No gradient overflow detected in the last {} step(s) after step {}, increasing loss scale from {:g} to {:g}.", self._fp16_loss_scaler.scale_window, self._step_nr, result.old_scale, result.new_scale)  # fmt: skip
        elif result.overflowed:
            log.info("Overflow detected at step {}, ignoring gradient, decreasing loss scale from {:g} to {:g}.", self._step_nr, result.old_scale, result.new_scale)  # fmt: skip

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

        self._maybe_save_checkpoint(blocking=False)

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
            self._maybe_save_checkpoint(blocking=False)

            self._validate()

            self._maybe_complete_checkpoint_save_operation()
        else:
            self._maybe_save_checkpoint(blocking=True)

        return _TrainerState.STOPPED

    def _early_stop(self) -> _TrainerState:
        log.info("Early stop requested. Stopping training at step {}.", self._step_nr)  # fmt: skip

        self._stop_requested = True

        should_save_checkpoint = self._should_save_checkpoint()
        if should_save_checkpoint:
            self._save_checkpoint(blocking=True)
        else:
            self._maybe_complete_checkpoint_save_operation()

        return _TrainerState.STOPPED

    def _maybe_save_checkpoint(self, blocking: bool) -> None:
        should_save_checkpoint = self._should_save_checkpoint()
        if should_save_checkpoint:
            self._save_checkpoint(blocking)

    def _should_save_checkpoint(self) -> bool:
        return self._should_do(
            self._checkpoint_after_n_steps,
            self._checkpoint_every_n_steps,
            self._checkpoint_after_n_data_epochs,
            self._checkpoint_every_n_data_epochs,
        )

    def _save_checkpoint(self, blocking: bool) -> None:
        step_nr = self._step_nr

        log.info("Preparing checkpoint at step {}.", step_nr)

        self._maybe_complete_checkpoint_save_operation()

        def on_checkpoint_ready(step_nr: int, blocking: bool) -> None:
            if blocking:
                log.info("Checkpoint prepared. Saving.")
            else:
                log.info("Checkpoint prepared. Saving asynchronously.")

        if isinstance(self._save_model_only, bool):
            save_model_only = self._save_model_only
        else:
            save_model_only = self._save_model_only == "all"

        if save_model_only:
            self._checkpoint_manager.save_model_only(
                step_nr,
                self._unit.model,
                ready_callback=on_checkpoint_ready,
                saved_callback=self._on_checkpoint_saved,
                blocking=blocking,
            )
        else:
            trainer = _TrainerCheckpointProxy(self)

            self._checkpoint_manager.save_checkpoint(
                step_nr,
                trainer,
                self._unit.model,
                self._optimizer,
                self._data_reader,
                ready_callback=on_checkpoint_ready,
                saved_callback=self._on_checkpoint_saved,
                blocking=blocking,
            )

    def _on_checkpoint_saved(self, step_nr: int, blocking: bool) -> None:
        log.info("Checkpoint at step {} saved.", step_nr)

        gangs = self._gangs

        hg_exporter = self._checkpoint_hg_exporter

        if hg_exporter is not NOOP_CHECKPOINT_HG_EXPORTER:
            if gangs.root.rank == 0:
                if hg_exporter.is_exporting:
                    log.info("Waiting for the current Hugging Face model export operation to complete before continuing.")  # fmt: skip

                hg_exporter.complete_pending()

            gangs.root.barrier()

        self._delete_stale_checkpoints()

        if self._save_model_only == "all_but_last":
            self._delete_previous_non_model_checkpoints()

        if hg_exporter is not NOOP_CHECKPOINT_HG_EXPORTER:
            if gangs.root.rank == 0:
                if blocking:
                    log.info("Exporting Hugging Face model of step {}.", step_nr)  # fmt: skip
                else:
                    log.info("Asynchronously exporting Hugging Face model of step {}.", step_nr)  # fmt: skip

                def on_exported(step_nr: int) -> None:
                    log.info("Hugging Face model of step {} exported.", step_nr)

                hg_exporter.export(
                    step_nr, exported_callback=on_exported, blocking=blocking
                )

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
        except GangError as ex:
            raise_operational_gang_error(ex)

        if gangs.root.rank == 0:
            if values is None:
                raise InternalError("`values` is `None`.")

            self._unit.process_metric_values(values)

            # If the optimizer has a single parameter group, report the learning
            # rate as a scalar.
            if len(self._last_lrs) == 1:
                values["lr"] = self._last_lrs[0]
            else:
                values["lr"] = self._last_lrs

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

            self._metric_recorder.record_metric_values("train", values, self._step_nr)

        gangs.root.barrier()

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
        if self._validator is NOOP_VALIDATOR:
            return False

        return self._should_do(
            self._validate_after_n_steps,
            self._validate_every_n_steps,
            self._validate_after_n_data_epochs,
            self._validate_every_n_data_epochs,
        )

    def _validate(self) -> float | None:
        if self._validator is NOOP_VALIDATOR:
            raise InternalError("`_validator` is noop.")

        if self._step_nr == 0:
            log.info("Starting pre-validation before training.")
        else:
            log.info("Starting validation after step {}.", self._step_nr)

        with self._unit.model.summon_full_parameters():
            score = self._validator.run(self._step_nr)

        if score is not None:
            if self._should_save_checkpoint():
                self._save_score(score)

        self._validator.reset()

        self._unit.model.module.train()

        # Try to avoid CUDA memory fragmentation after validation.
        if self._gangs.root.device.type == "cuda":
            torch.cuda.empty_cache()

        log.info("Validation finished.")

        return score

    def _save_score(self, score: float) -> None:
        self._checkpoint_manager.save_score(self._step_nr, score)

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

    def _maybe_complete_checkpoint_save_operation(self) -> None:
        if not self._checkpoint_manager.is_saving:
            return

        log.info("Waiting for the current checkpoint save operation to complete before continuing.")  # fmt: skip

        self._checkpoint_manager.maybe_complete_save_operation(blocking=True)

    def _delete_previous_non_model_checkpoints(self) -> None:
        step_nrs = self._checkpoint_manager.get_step_numbers(exclude_model_only=True)

        num_steps = len(step_nrs)
        if num_steps <= 1:
            return

        log.info("Deleting non-model state of previous {} checkpoint(s).", num_steps - 1)  # fmt: skip

        for step_nr in step_nrs[:-1]:  # always keep the last checkpoint.
            self._checkpoint_manager.delete_checkpoint(step_nr, keep_model=True)

        log.info("Non-model state of previous checkpoints deleted.")

    def _delete_stale_checkpoints(self) -> None:
        stale_step_nrs = self._checkpoint_manager.get_stale_step_numbers(
            self._keep_last_n_checkpoints,
            self._keep_best_n_checkpoints,
            self._keep_checkpoint_every_n_steps,
        )

        if not stale_step_nrs:
            return

        log.info("Deleting {} stale checkpoint(s).", len(stale_step_nrs))

        for step_nr in stale_step_nrs:
            self._checkpoint_manager.delete_checkpoint(step_nr)

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


class _TrainerCheckpointProxy(Stateful):
    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer

    @override
    def state_dict(self) -> dict[str, object]:
        wall_time = self._trainer._wall_watch.get_elapsed_time()

        base_wall_time = self._trainer._base_wall_time + wall_time

        state_dict: dict[str, object] = {
            "_step_nr": self._trainer._step_nr,
            "_data_epoch_nr": self._trainer._data_epoch_nr,
            "_lr_scheduler": self._trainer._lr_scheduler.state_dict(),
            "_fp16_loss_scaler": self._trainer._fp16_loss_scaler.state_dict(),
            "_rng_bag": self._trainer._rng_bag.state_dict(),
            "_metric_bag": self._trainer._metric_bag.state_dict(),
            "_base_wall_time": base_wall_time,
        }

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        state_dict = dict(state_dict)

        def get_state(name: str, kls: type[T]) -> T:
            try:
                state = state_dict.pop(name)
            except KeyError:
                raise StateDictError(
                    f"`state_dict` is expected to contain a key named '{name}'."
                )

            if not isinstance(state, kls):
                raise StateDictError(
                    f"`state_dict['{name}']` is expected to be of type `{kls}`, but is of type `{type(state)}` instead."
                )

            return state

        def load_stateful(name: str, obj: Stateful) -> None:
            attr_state_dict = get_state(name, dict)

            try:
                obj.load_state_dict(attr_state_dict)
            except (RuntimeError, ValueError, TypeError, StateDictError) as ex:
                raise StateDictError(
                    f"`state_dict['{name}']` does not represent a valid `{type(obj)}` state."
                ) from ex

        self._trainer._step_nr = get_state("_step_nr", int)
        self._trainer._data_epoch_nr = get_state("_data_epoch_nr", int)

        load_stateful("_lr_scheduler", self._trainer._lr_scheduler)
        load_stateful("_rng_bag", self._trainer._rng_bag)
        load_stateful("_metric_bag", self._trainer._metric_bag)
        load_stateful("_fp16_loss_scaler", self._trainer._fp16_loss_scaler)

        self._trainer._base_wall_time = get_state("_base_wall_time", float)

        StateDictError.raise_if_not_empty(state_dict)
