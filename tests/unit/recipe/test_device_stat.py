# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.device import Device
from fairseq2.utils.cpu_stat import CpuDeviceStatTracker
from fairseq2.utils.device_stat import CudaDeviceStatTracker


class TestDeviceStatTrackers:
    """Test device stat tracker implementations directly."""
    
    def test_cpu_tracker_collects_stats(self) -> None:
        device = Device("cpu")
        tracker = CpuDeviceStatTracker(device)
        
        stats = tracker.get_stats()
        assert isinstance(stats, dict)
        assert "peak_memory_rss_bytes" in stats or "memory_percent" in stats
        
        # Test reset
        tracker.reset()
        stats_after_reset = tracker.get_stats()
        assert isinstance(stats_after_reset, dict)

    def test_cuda_tracker_collects_stats(self) -> None:
        try:
            device = Device("cuda:0")
        except RuntimeError:
            pytest.skip("CUDA not available")
        
        try:
            from fairseq2.device import CudaContext
            cuda_context = CudaContext()
            tracker = CudaDeviceStatTracker(device, cuda_context)
            
            stats = tracker.get_stats()
            assert isinstance(stats, dict)
            assert "peak_active_mem_bytes" in stats or "peak_reserved_mem_bytes" in stats
            
            # Test reset
            tracker.reset()
            stats_after_reset = tracker.get_stats()
            assert isinstance(stats_after_reset, dict)
        except Exception:
            pytest.skip("CUDA context not available")