# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import OrderedDict
from signal import SIGUSR1, signal
from types import FrameType
from typing import Protocol, final

from torch.utils.hooks import RemovableHandle
from typing_extensions import override

from fairseq2.error import OperationalError, InternalError
from fairseq2.evaluator import Evaluator, EvaluatorError
from fairseq2.gang import Gangs
from fairseq2.generator import Generator, GeneratorError
from fairseq2.logging import log
from fairseq2.recipe.error import RecipeError
from fairseq2.recipe.task import Task
from fairseq2.trainer import (
    CorruptCheckpointError,
    MinimumLossScaleReachedError,
    Trainer,
    TrainError,
)
from fairseq2.utils.stopwatch import Stopwatch


class _TaskStartHook(Protocol):
    def __call__(self) -> None: ...


@final
class _TaskRunner:
    def __init__(self, gangs: Gangs, wall_watch: Stopwatch) -> None:
        self._gangs = gangs
        self._wall_watch = wall_watch
        self._start_hooks: dict[int, _TaskStartHook] = OrderedDict()

    def run(self, task: Task) -> None:
        log.info("Running on {} process(es).", self._gangs.root.size)

        # Use SIGUSR1 as the stop signal.
        def request_stop(signum: int, frame: FrameType | None) -> None:
            log.info("SIGUSR1 received. Requesting recipe to stop.")

            task.request_stop()

        original_signal_handler = signal(SIGUSR1, request_stop)

        for hook in self._start_hooks.values():
            hook()

        try:
            done = task.run()
        except KeyboardInterrupt:
            elapsed_time = int(self._wall_watch.get_elapsed_time())

            if task.step_nr == 0:
                log.info("Task canceled after {:,} second(s)!", elapsed_time)
            else:
                log.info("Task canceled after {:,} second(s) at step {}!", elapsed_time, task.step_nr)  # fmt: skip

            raise
        else:
            elapsed_time = int(self._wall_watch.get_elapsed_time())

            if done:
                if task.step_nr == 0:
                    log.info("Task finished in {:,} second(s)!", elapsed_time)
                else:
                    log.info("Task finished in {:,} second(s) after {} step(s)!", elapsed_time, task.step_nr)  # fmt: skip
            else:
                if task.step_nr == 0:
                    log.info("Task stopped after {:,} second(s)!", elapsed_time)
                else:
                    log.info("Task stopped after {:,} second(s) at step {}!", elapsed_time, task.step_nr)  # fmt: skip
        finally:
            task.close()

            signal(SIGUSR1, original_signal_handler)

    def register_start_hook(self, hook: _TaskStartHook) -> RemovableHandle:
        handle = RemovableHandle(self._start_hooks)

        self._start_hooks[handle.id] = hook

        return handle


@final
class _TrainTask(Task):
    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer

    @override
    def run(self) -> bool:
        try:
            return self._trainer.run()
        except CorruptBatchError as ex:
            raise RecipeError(
                f"A corrupt data batch encountered at step {ex.step_nr}."
            ) from ex
        except CorruptCheckpointError as ex:
            raise RecipeError(
                f"Training state cannot be restored. Checkpoint of last step (step {ex.step_nr}) is corrupt."
            ) from ex
        except MinimumLossScaleReachedError as ex:
            raise RecipeError(
                f"Overflow detected at step {ex.step_nr}, loss scale is already at minimum ({ex.loss_scale:g}). Loss is probably exploding. Try lowering the learning rate, using gradient clipping, or increasing the batch size."
            ) from None
        except ValidationError as ex:
            match cause := ex.__cause__:
                case CorruptBatchError():
                    if ex.step_nr == 0:
                        raise RecipeError(
                            f"Pre-Validation before training failed. A corrupt data batch encountered at validation step {cause.step_nr}."
                        ) from ex
                    else:
                        raise RecipeError(
                            f"Validation after step {ex.step_nr} failed. A corrupt data batch encountered at validation step {cause.step_nr}."
                        ) from ex
                case _:
                    raise InternalError(f"Unexpected error of type `{type(cause)}`.")
        except TrainError as ex:
            raise OperationalError("Training failed.") from ex

    @override
    def request_stop(self) -> None:
        self._trainer.request_stop()

    @override
    def close(self) -> None:
        self._trainer.close()

    @property
    @override
    def step_nr(self) -> int:
        return self._trainer.step_nr


@final
class _EvalTask(Task):
    def __init__(self, evaluator: Evaluator) -> None:
        self._evaluator = evaluator

    @override
    def run(self) -> bool:
        return self._evaluator.run()

    @override
    def request_stop(self) -> None:
        self._evaluator.request_stop()

    @override
    def close(self) -> None:
        self._evaluator.close()

    @property
    @override
    def step_nr(self) -> int:
        return self._evaluator.step_nr


@final
class _GenerationTask(Task):
    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    @override
    def run(self) -> bool:
        return self._generator.run()

    @override
    def request_stop(self) -> None:
        self._generator.request_stop()

    @override
    def close(self) -> None:
        self._generator.close()

    @property
    @override
    def step_nr(self) -> int:
        return self._generator.step_nr
