# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.recipe.internal.task import _TaskRunner
from fairseq2.trainer import Trainer


@final
class _TrainHookManager:
    def maybe_register_trainer_hooks(self, trainer: Trainer) -> None:
        pass


@final
class _HookManager:
    def maybe_register_task_hooks(self, task_runner: _TaskRunner) -> None:
        pass
