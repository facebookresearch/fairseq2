# Copyright (c) Meta Platforms, Inc. and affiliates.e
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import final

from fairseq2.gang import Gangs
from fairseq2.profilers import TorchProfiler
from fairseq2.recipe.config import CommonSection


@final
class _MaybeTorchProfilerFactory:
    def __init__(self, section: CommonSection, output_dir: Path, gangs: Gangs) -> None:
        self._section = section
        self._output_dir = output_dir
        self._gangs = gangs

    def maybe_create(self) -> TorchProfiler | None:
        tp_config = self._section.profilers.torch

        if not tp_config.enabled:
            return None

        return TorchProfiler(
            tp_config.skip_n_steps,
            tp_config.wait_n_steps,
            tp_config.num_warmup_steps,
            tp_config.num_active_steps,
            tp_config.repeat,
            self._output_dir,
            self._gangs,
        )
