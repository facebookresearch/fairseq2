# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from itertools import count
from pathlib import Path
from typing import (
    Any,
    ContextManager,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    final,
)

import torch
import torch.distributed
from rich.progress import Progress
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.profiler import record_function
from torcheval.metrics import Mean

from fairseq2.checkpoint import CheckpointManager, CheckpointNotFoundError
from fairseq2.datasets import DataReader
from fairseq2.gang import FakeGang, Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    LogMetricRecorder,
    MetricBag,
    MetricRecorder,
    TensorBoardRecorder,
    merge_metric_states,
    record_metrics,
)
from fairseq2.nn.utils.gradient import (
    check_gradient_norms,
    clip_gradient_norm,
    normalize_gradients,
)
from fairseq2.optim import DynamicLossScaler
from fairseq2.optim.lr_scheduler import LRScheduler, NoopLR, get_effective_lr
from fairseq2.recipes.criterion import Criterion
from fairseq2.recipes.utils.cli import create_rich_progress
from fairseq2.typing import CPU, DataType, override
from fairseq2.utils.profiler import Profiler, Stopwatch
from fairseq2.utils.rng import RngBag
from fairseq2.utils.state import FSDPOptimizerStateHandler, StatefulObjectBag

log = get_log_writer(__name__)


class Trainer(ABC):
    """Trains a machine learning model."""

    @abstractmethod
    def __call__(self) -> None:
        """Run training."""


BatchT = TypeVar("BatchT")


@final
class StandardTrainer(StatefulObjectBag, Trainer, Generic[BatchT]):
    """Trains a machine learning model with a common set of features."""

    _model: Module
    _criterion: Criterion[BatchT]
    _root_gang: Gang
    _dp_gang: Gang
    _tp_gang: Gang
    _dtype: DataType
    _optimizer: Optimizer
    _lr_scheduler: LRScheduler
    _loss_scaler: DynamicLossScaler
    _max_gradient_norm: Optional[float]
    _train_data_reader: DataReader[BatchT]
    _valid_data_readers: Dict[str, DataReader[BatchT]]
    _step_nr: int
    _max_num_steps: Optional[int]
    _data_epoch_nr: int
    _max_num_data_epochs: Optional[int]
    _eod: bool
    _validate_after_n_steps: int
    _validate_every_n_steps: int
    _checkpoint_manager: CheckpointManager
    _checkpoint_after_n_steps: int
    _checkpoint_every_n_steps: Optional[int]
    _keep_last_n_checkpoints: Optional[int]
    _keep_best_n_checkpoints: Optional[int]
    _train_metric_bag: MetricBag
    _valid_metric_bag: MetricBag
    _metric_recorders: List[MetricRecorder]
    _publish_metrics_after_n_steps: int
    _publish_metrics_every_n_steps: int
    _profiler: Profiler
    _anomaly_detection: bool
    _seed: int
    _rng_bag: RngBag
    _wall_watch: Stopwatch
    _train_elapsed_time: float
    _valid_elapsed_time: float
    _run: bool
    _progress: Progress

    def __init__(
        self,
        *,
        criterion: Criterion[BatchT],
        gang: Gang,
        optimizer: Optimizer,
        data_readers: Mapping[str, DataReader[BatchT]],
        checkpoint_manager: CheckpointManager,
        wall_watch: Stopwatch,
        dtype: DataType = torch.float32,
        dp_gang: Optional[Gang] = None,
        tp_gang: Optional[Gang] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        fp16_loss_scale: Tuple[float, float] = (128.0, 0.0001),
        max_gradient_norm: Optional[float] = None,
        max_num_steps: Optional[int] = 1000,
        max_num_data_epochs: Optional[int] = None,
        validate_after_n_steps: int = 0,
        validate_every_n_steps: int = 100,
        checkpoint_after_n_steps: int = 0,
        checkpoint_every_n_steps: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        keep_best_n_checkpoints: Optional[int] = None,
        tb_dir: Optional[Path] = None,
        publish_metrics_after_n_steps: int = 0,
        publish_metrics_every_n_steps: int = 100,
        profile: Optional[Tuple[int, int]] = None,
        anomaly_detection: bool = False,
        seed: int = 2,
    ) -> None:
        """
        :param criterion:
            The criterion for loss computation.
        :param gang:
            The gang to use for distributed training.
        :param optimizer:
            The parameter optimizer.
        :param data_readers:
            The split-data reader pairs. Must contain at least a data reader for
            the 'train' split.
        :param checkpoint_manager:
            The checkpoint manager.
        :param wall_watch:
            The stopwatch to track process wall-time.
        :param dtype:
            The data type to train with.
        :param dp_gang:
            The data parallel gang. If ``None``, ``gang`` will be used.
        :param tp_gang:
            The tensor parallel gang. Only required for tensor parallel models
            such as LLaMA 70B.
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
        :param validate_after_n_steps:
            The number of steps after which to start validating the model.
        :param validate_every_n_steps:
            The step interval at which to validate the model.
        :param checkpoint_after_n_steps:
            The number of steps after which to start checkpointing.
        :param checkpoint_every_n_steps:
            The step interval at which to checkpoint.
        :param keep_last_n_checkpoints:
            The number of checkpoints to keep. If ``None``, no checkpoint will
            be deleted.
        :param keep_best_n_checkpoints:
            WIP
        :param tb_dir:
            The TensorBoard log directory to dump metrics.
        :param publish_metrics_after_n_steps:
            The number of steps after which to start publishing metrics.
        :param publish_metrics_every_n_steps:
            The step interval at which to publish metrics.
        :param profile:
            The number of steps that the PyTorch profiler should skip and then
            record.
        :param anomaly_detection:
            If ``True``, turns on anomaly detection feature in ``torch.autograd``.
        :param seed:
            The random number generator seed to use.
        """
        super().__init__()

        device = gang.device

        self._model = criterion.model

        criterion.train_metric_bag.register_metric(
            "gradient_norm", Mean(device=device), persistent=False
        )

        self._criterion = criterion

        self._root_gang = gang

        if dp_gang is not None and tp_gang is not None:
            self._dp_gang = dp_gang
            self._tp_gang = tp_gang
        elif dp_gang is None and tp_gang is None:
            self._dp_gang = gang
            self._tp_gang = FakeGang(device=device)
        else:
            raise ValueError("`dp_gang` and `tp_gang` must be both specified.")

        self._dtype = dtype

        if uses_fsdp := isinstance(self._model, FSDP):
            self.register_stateful(
                "_optimizer", optimizer, FSDPOptimizerStateHandler(self._model)
            )
        else:
            self._optimizer = optimizer

        self._lr_scheduler = lr_scheduler or NoopLR(optimizer)

        self._valid_data_readers = dict(data_readers)

        try:
            self._train_data_reader = self._valid_data_readers.pop("train")
        except KeyError:
            raise ValueError(
                "`data_readers` must contain at least a data reader for the 'train' split."
            )

        fp16_init_scale, fp16_min_scale = fp16_loss_scale

        self._loss_scaler = DynamicLossScaler(
            optimizer,
            gang,
            sharded=uses_fsdp or self._tp_gang.size > 0,
            init_scale=fp16_init_scale,
            min_scale=fp16_min_scale,
            gradient_accumulation=self._train_data_reader.num_accumulate,
            enabled=self._dtype == torch.float16,
        )

        self._max_gradient_norm = max_gradient_norm

        self.register_stateful("_step_nr", 0)

        self._max_num_steps = max_num_steps

        self.register_stateful("_data_epoch_nr", 1)

        self._max_num_data_epochs = max_num_data_epochs

        self._eod = max_num_data_epochs == 0

        if validate_every_n_steps == 0:
            raise ValueError("`validate_every_n_steps` must be greater than zero.")

        self._validate_after_n_steps = validate_after_n_steps
        self._validate_every_n_steps = validate_every_n_steps

        self._checkpoint_manager = checkpoint_manager

        if checkpoint_every_n_steps == 0:
            raise ValueError("`checkpoint_every_n_steps` must be greater than zero.")

        self._checkpoint_after_n_steps = checkpoint_after_n_steps
        self._checkpoint_every_n_steps = checkpoint_every_n_steps

        if keep_last_n_checkpoints == 0:
            raise ValueError("`keep_last_n_checkpoints` must be greater than zero.")

        if keep_best_n_checkpoints == 0:
            raise ValueError("`keep_best_n_checkpoints` must be greater than zero.")

        self._keep_last_n_checkpoints = keep_last_n_checkpoints
        self._keep_best_n_checkpoints = keep_best_n_checkpoints

        self._train_metric_bag = self._criterion.train_metric_bag

        self.register_non_stateful(
            "_valid_metric_bag", self._criterion.valid_metric_bag
        )

        if self._tp_gang.rank == 0 and self._dp_gang.rank == 0:
            self._metric_recorders = [LogMetricRecorder(log)]

            if tb_dir is not None:
                self._metric_recorders.append(TensorBoardRecorder(tb_dir))
        else:
            self._metric_recorders = []

        if publish_metrics_every_n_steps == 0:
            raise ValueError(
                "`publish_metrics_every_n_steps` must be greater than zero."
            )

        self._publish_metrics_after_n_steps = publish_metrics_after_n_steps
        self._publish_metrics_every_n_steps = publish_metrics_every_n_steps

        if profile is None or tb_dir is None:
            skip_first, active_steps = 1, 0

            if tb_dir is None:
                log.warning("No TensorBoard log directory provided. Profiling will be disabled.")  # fmt: skip

            profile_dir = Path()
        else:
            skip_first, active_steps = profile

            profile_dir = tb_dir

        self._profiler = Profiler(
            skip_first, active_steps, profile_dir, gang, enabled=active_steps > 0
        )

        self._anomaly_detection = anomaly_detection

        self._seed = seed

        self._rng_bag = RngBag.from_device_defaults(CPU, device)

        self._wall_watch = wall_watch

        self._train_elapsed_time = 0.0
        self._valid_elapsed_time = 0.0

        self._run = False

        self._progress = create_rich_progress()

    @override
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

        elapsed_time = self._wall_watch.get_elapsed_time()

        log.info("Training complete in {:,} seconds after {} steps!", int(elapsed_time), self._step_nr)  # fmt: skip

    def _maybe_restore_state(self) -> None:
        log.info("Attempting to load the last checkpoint.")

        try:
            step_nr, checkpoint = self._checkpoint_manager.load_last_checkpoint()
        except CheckpointNotFoundError:
            log.info("No checkpoint found. Starting training.")

            return

        log.info("Checkpoint loaded, restoring training from step {}.", step_nr)

        self._step_nr = step_nr

        self.load_state_dict(checkpoint)

        self._root_gang.barrier()

        log.info("Training restored, resuming.")

    def _do_run(self) -> None:
        with self._progress, self._profiler:
            train_task = self._progress.add_task(
                "train", total=self._max_num_steps, completed=self._step_nr
            )

            while self._should_step():
                self._step_nr += 1

                self._progress.update(train_task, refresh=True, advance=1)

                detect_anomaly = torch.autograd.set_detect_anomaly(  # type: ignore[attr-defined]
                    self._anomaly_detection, check_nan=True
                )

                with detect_anomaly:
                    with record_function(f"step_{self._step_nr}"):
                        try:
                            self._run_train_step()
                        except StopIteration:
                            self._eod = True

                if self._should_publish_train_metrics():
                    self._publish_train_metrics()

                if self._should_checkpoint():
                    self._checkpoint()

                if self._should_validate():
                    self._validate()

                self._profiler.step()

    def _should_step(self) -> bool:
        if self._eod:
            return False

        if self._max_num_steps is None:
            return True

        return self._step_nr < self._max_num_steps

    def _run_train_step(self) -> bool:
        step_nr = self._step_nr

        log.debug("Running training step {}.", step_nr)

        stepped = False

        watch = Stopwatch(start=True, device=self._root_gang.device)

        with record_function(f"step_{step_nr}_prologue"):
            self._criterion.set_step(step_nr)

        while not stepped:
            # Collect the batches.
            with record_function(f"step_{step_nr}_data_load"):
                batches = self._next_train_batches()

            num_targets = 0

            self._train_metric_bag.begin_updates()

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
                self._train_metric_bag.rollback_updates()

                if scale_result.min_reached:
                    raise FloatingPointError(
                        f"The gradients are scaled down to minimum at step {step_nr}. Training cannot continue."
                    )

                log.debug("Repeating training step {}.", step_nr)
            else:
                self._lr_scheduler.step()

                stepped = True

            # Reset.
            self._optimizer.zero_grad(set_to_none=True)

        self._train_metric_bag.commit_updates()

        self._train_metric_bag.gradient_norm.update(grad_norm)

        self._train_elapsed_time += watch.get_elapsed_time()

        return True

    def _next_train_batches(self) -> List[BatchT]:
        while True:
            try:
                return next(self._train_data_reader)
            except StopIteration:
                log.info("End of epoch {} reached at training step {}.", self._data_epoch_nr, self._step_nr)  # fmt: skip

                if self._max_num_data_epochs is not None:
                    if self._data_epoch_nr >= self._max_num_data_epochs:
                        break

                self._data_epoch_nr += 1

            self._train_data_reader.reset()

        log.info("End of data reached at training step {}.", self._step_nr)

        raise StopIteration()

    def _maybe_no_sync(self, batch_nr: int, num_batches: int) -> ContextManager[None]:
        if batch_nr < num_batches - 1 and self._dp_gang.size > 1:
            return self._model.no_sync()  # type: ignore[no-any-return]

        return nullcontext()

    def _compute_loss(self, batch: BatchT) -> Tuple[Tensor, int]:
        with self._maybe_autocast():
            return self._criterion.compute_loss(batch)

    def _maybe_autocast(self) -> ContextManager[None]:
        if self._dtype == torch.float32:
            return nullcontext()

        if isinstance(self._model, (DDP, FSDP)):
            if self._model.mixed_precision is not None:
                return nullcontext()

        return torch.autocast(device_type=self._dp_gang.device.type, dtype=self._dtype)

    def _should_publish_train_metrics(self) -> bool:
        if self._step_nr < self._publish_metrics_after_n_steps:
            return False

        return self._should_do(self._publish_metrics_every_n_steps)

    def _publish_train_metrics(self) -> None:
        log.debug("Syncing train metrics.")

        if self._tp_gang.rank == 0:
            values = self._train_metric_bag.sync_and_compute_metrics()
        else:
            values = None

        self._train_metric_bag.reset_non_persistent_metrics()

        elapsed_time = self._train_elapsed_time

        self._train_elapsed_time = 0.0

        if self._tp_gang.rank != 0 or self._dp_gang.rank != 0:
            return

        assert values is not None

        values["lr"] = get_effective_lr(self._lr_scheduler)

        self._set_throughput(values, elapsed_time)

        values["elapsed_time"] = elapsed_time

        values["wall_time"] = self._wall_watch.get_elapsed_time()

        record_metrics(self._metric_recorders, "train", values, self._step_nr)

    def _should_validate(self) -> bool:
        if not self._valid_data_readers:
            return False

        if self._step_nr < self._validate_after_n_steps:
            return False

        return self._should_do(self._validate_every_n_steps)

    @torch.inference_mode()
    def _validate(self) -> None:
        self._model.eval()

        log.info("Starting validation after step {}.", self._step_nr)

        if len(self._valid_data_readers) == 1:
            for data_reader in self._valid_data_readers.values():
                self._do_validate(data_reader)
        else:
            elapsed_time = 0.0

            cumulative_metrics = deepcopy(self._valid_metric_bag.metrics)

            for split, data_reader in self._valid_data_readers.items():
                log.info("Validating split '{}'.", split)

                self._do_validate(data_reader)

                elapsed_time += self._valid_elapsed_time

                merge_metric_states(self._valid_metric_bag.metrics, cumulative_metrics)

                self._publish_validation_metrics(split)

            self._valid_elapsed_time = elapsed_time

            merge_metric_states(cumulative_metrics, self._valid_metric_bag.metrics)

        self._publish_validation_metrics()

        log.info("Validation complete, resuming training.")

        self._model.train()

    def _do_validate(self, data_reader: DataReader[BatchT]) -> None:
        valid_task = self._progress.add_task("valid", total=None, completed=0)

        for step_nr in count(start=1):
            self._progress.update(valid_task, refresh=True, advance=1)

            log.debug("Running validation step {}.", step_nr)

            watch = Stopwatch(start=True, device=self._root_gang.device)

            try:
                batches = next(data_reader)
            except StopIteration:
                break

            for batch in batches:
                self._compute_loss(batch)

            self._valid_elapsed_time += watch.get_elapsed_time()

        self._progress.remove_task(valid_task)

        data_reader.reset()

    def _publish_validation_metrics(self, split: Optional[str] = None) -> None:
        log.debug("Syncing validation metrics.")

        if self._tp_gang.rank == 0:
            values = self._valid_metric_bag.sync_and_compute_metrics()
        else:
            values = None

        self._valid_metric_bag.reset_metrics()

        elapsed_time = self._valid_elapsed_time

        self._valid_elapsed_time = 0.0

        if self._tp_gang.rank != 0 or self._dp_gang.rank != 0:
            return

        assert values is not None

        self._set_throughput(values, elapsed_time)

        values["elapsed_time"] = elapsed_time

        values["wall_time"] = self._wall_watch.get_elapsed_time()

        run = "valid"

        if split:
            run = f"{run}:{split}"

        record_metrics(self._metric_recorders, run, values, self._step_nr)

    def _set_throughput(
        self, metric_values: Dict[str, Any], elapsed_time: float
    ) -> None:
        try:
            num_elements = metric_values[self._criterion.throughput_metric_name]
        except KeyError:
            return

        if not isinstance(num_elements, (int, float, Tensor)):
            return

        if elapsed_time == 0.0:
            metric_values["elements_per_second"] = 0.0
        else:
            metric_values["elements_per_second"] = num_elements / elapsed_time

    def _should_checkpoint(self) -> bool:
        if self._step_nr < self._checkpoint_after_n_steps:
            return False

        if n := self._checkpoint_every_n_steps:
            return self._should_do(n)

        return False

    def _checkpoint(self) -> None:
        step_nr = self._step_nr

        log.info("Saving checkpoint after step {}.", step_nr)

        checkpoint = self.state_dict()

        log.info("State dictionary of the trainer extracted.")

        if self._dp_gang.size > 1 and isinstance(self._model, DDP):
            replicated_keys = {"_model", "_optimizer"}
        else:
            replicated_keys = None

        self._checkpoint_manager.save_checkpoint(
            step_nr, checkpoint, model_key="_model", replicated_keys=replicated_keys
        )

        log.info("Checkpoint saved.")

        if isinstance(self._model, FSDP):
            log.info("Saving consolidated FSDP model after step {}.", step_nr)

            self._checkpoint_manager.save_consolidated_fsdp_model(step_nr, self._model)

            log.info("Model saved.")

        if n := self._keep_last_n_checkpoints:
            self._checkpoint_manager.keep_last_n_checkpoints(n)

    def _should_do(self, n_step: int) -> bool:
        if self._eod:
            return True

        if self._max_num_steps and self._step_nr >= self._max_num_steps:
            return True

        return self._step_nr % n_step == 0
