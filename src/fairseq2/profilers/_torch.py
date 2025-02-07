# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from typing_extensions import override

from fairseq2.gang import Gangs
from fairseq2.profilers._handler import ProfilerHandler
from fairseq2.profilers._profiler import AbstractProfiler, NoopProfiler, Profiler
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@final
class TorchProfiler(AbstractProfiler):
    """Represents a convenience wrapper for :class:`profile`."""

    _profile: profile

    def __init__(
        self,
        skip_n_steps: int,
        wait_n_steps: int,
        num_warmup_steps: int,
        num_active_steps: int,
        repeat: int,
        log_dir: Path,
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


TORCH_PROFILER: Final = "torch"


@dataclass(kw_only=True)
class TorchProfilerConfig:
    enabled: bool = False

    skip_n_steps: int = 4

    wait_n_steps: int = 0

    num_warmup_steps: int = 1

    num_active_steps: int = 4

    repeat: int = 1


@final
class TorchProfilerHandler(ProfilerHandler):
    @override
    def create(self, config: object, gangs: Gangs, output_dir: Path) -> Profiler:
        config = structure(config, TorchProfilerConfig)

        validate(config)

        if not config.enabled:
            return NoopProfiler()

        log_dir = output_dir.joinpath("tb")

        return TorchProfiler(
            config.skip_n_steps,
            config.wait_n_steps,
            config.num_warmup_steps,
            config.num_active_steps,
            config.repeat,
            log_dir,
            gangs,
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return TorchProfilerConfig
