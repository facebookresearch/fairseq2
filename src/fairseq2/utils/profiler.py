# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from fairseq2.data.typing import PathLike
from fairseq2.gang import Gang


class Profiler:
    """Represents a convenience wrapper for :class:`profile`."""

    _profile: Optional[profile]

    def __init__(
        self,
        skip_first: int,
        active: int,
        log_dir: PathLike,
        gang: Gang,
        enabled: bool = False,
    ) -> None:
        """
        :param skip_first:
            The number of steps to skip at the beginning. The last skipped step
            will be treated as the warm-up step.
        :param active:
            The number of steps with active recording.
        :param log_dir:
            The TensorBoard log directory under which to store the trace files.
        :param gang:
            The associated gang.
        :param enabled:
            If ``False``, skips recording and becomes a no-op.
        """
        if skip_first <= 0:
            raise ValueError("`skip_first` must be greater than zero.")

        if not enabled:
            self._profile = None

            return

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        schedule_ = schedule(
            skip_first=skip_first - 1, wait=0, warmup=1, active=active, repeat=1
        )

        trace_handler = tensorboard_trace_handler(
            str(log_dir), worker_name=f"rank_{gang.rank}"
        )

        self._profile = profile(
            activities=activities,
            schedule=schedule_,
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
        )

    def start(self) -> None:
        """Start profiling."""
        if self._profile is not None:
            self._profile.start()

    def stop(self) -> None:
        """Stop profiling."""
        if self._profile is not None:
            self._profile.stop()

    def step(self) -> None:
        """Signal the profiler that the next profiling step has started."""
        if self._profile is not None:
            self._profile.step()

    def __enter__(self) -> "Profiler":
        self.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self.stop()

    @property
    def wrapped_profile(self) -> Optional[profile]:
        """The wrapped :class:`profile` instance."""
        return self._profile
