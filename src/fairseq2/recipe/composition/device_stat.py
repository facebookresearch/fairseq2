# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.internal.device_stat import _RecipeDeviceStatTrackerProvider
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.device_stat import (
    NOOP_DEVICE_STAT_TRACKER,
    CudaDeviceStatTracker,
    DeviceStatTracker,
)


def _register_device_stat(container: DependencyContainer) -> None:
    # Tracker
    def get_device_stat_tracker(resolver: DependencyResolver) -> DeviceStatTracker:
        tracker_provider = resolver.resolve(_RecipeDeviceStatTrackerProvider)

        return tracker_provider.get()

    container.register(DeviceStatTracker, get_device_stat_tracker, singleton=True)

    container.register_type(_RecipeDeviceStatTrackerProvider)

    # CUDA
    container.register_type(DeviceStatTracker, CudaDeviceStatTracker, key="cuda")

    # CPU
    # No-op implementation is enough for now, since CPU-only is not a primary use case
    container.register_instance(DeviceStatTracker, NOOP_DEVICE_STAT_TRACKER, key="cpu")
