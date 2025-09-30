# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import final

from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from typing_extensions import override

from fairseq2.gang import Gangs
from fairseq2.profilers.profiler import Profiler


@final
class TorchProfiler(Profiler):
    """Represents a convenience wrapper for :class:`profile`."""

    def __init__(
        self,
        skip_n_steps: int,
        wait_n_steps: int,
        num_warmup_steps: int,
        num_active_steps: int,
        repeat: int,
        output_dir: Path,
        gangs: Gangs,
    ) -> None:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        schedule_ = schedule(
            skip_first=skip_n_steps,
            wait=wait_n_steps,
            warmup=num_warmup_steps,
            active=num_active_steps,
            repeat=repeat,
        )

        log_dir = output_dir.joinpath("tb")

        trace_handler = tensorboard_trace_handler(
            str(log_dir), worker_name=f"rank_{gangs.root.rank}", use_gzip=True
        )

        self._profile = profile(
            activities=activities,
            schedule=schedule_,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

    @override
    def start(self) -> None:
        if self._profile is not None:
            self._profile.start()

    @override
    def stop(self) -> None:
        if self._profile is not None:
            self._profile.stop()

    @override
    def step(self) -> None:
        if self._profile is not None:
            self._profile.step()
