# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.device import Device
from fairseq2.error import OperationalError
from fairseq2.utils.cpu_stat import CpuDeviceStatTracker


class TestCpuDeviceStatTracker:
    def test_init_validates_device_type(self) -> None:
        with pytest.raises(ValueError, match=r"`device.type` must be `cpu`"):
            CpuDeviceStatTracker(Device("cuda"))

    def test_get_stats_returns_valid_metrics(self) -> None:
        tracker = CpuDeviceStatTracker(Device("cpu"))

        stats = tracker.get_stats()

        assert "peak_memory_rss_bytes" in stats
        assert "peak_memory_vms_bytes" in stats
        assert "memory_percent" in stats
        assert "num_threads" in stats

        assert stats["peak_memory_rss_bytes"] > 0
        assert stats["peak_memory_vms_bytes"] > 0
        assert 0 <= stats["memory_percent"] <= 100
        assert stats["num_threads"] > 0

    def test_get_stats_tracks_peak_memory(self) -> None:
        tracker = CpuDeviceStatTracker(Device("cpu"))

        stats1 = tracker.get_stats()
        peak1 = stats1["peak_memory_rss_bytes"]

        _ = [0] * 10000

        stats2 = tracker.get_stats()
        peak2 = stats2["peak_memory_rss_bytes"]

        assert peak2 >= peak1

    def test_reset_clears_peak_memory(self) -> None:
        tracker = CpuDeviceStatTracker(Device("cpu"))

        _ = tracker.get_stats()

        tracker.reset()

        stats = tracker.get_stats()
        assert stats["peak_memory_rss_bytes"] >= 0
        assert stats["peak_memory_vms_bytes"] >= 0
 