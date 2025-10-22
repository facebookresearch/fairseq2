#!/usr/bin/env python3
""" 
DEVELOPMENT ONLY FILE

Simple test script to verify CPU stat tracker implementation.
Run this after installing fairseq2 with: pip install -e .
"""

from fairseq2.device import Device
from fairseq2.utils.cpu_stat import CpuDeviceStatTracker


def test_cpu_tracker():
    print("Testing CpuDeviceStatTracker...")
    
    device = Device("cpu")
    tracker = CpuDeviceStatTracker(device)
    
    print("\n1. Getting initial stats...")
    stats = tracker.get_stats()
    
    print("Stats collected:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n2. Allocating some memory...")
    data = [0] * 1000000
    
    print("\n3. Getting stats after allocation...")
    stats_after = tracker.get_stats()
    
    print("Stats after allocation:")
    for key, value in stats_after.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n4. Verifying peak memory increased...")
    assert stats_after["peak_memory_rss_bytes"] >= stats["peak_memory_rss_bytes"]
    print("✓ Peak memory tracking works!")
    
    print("\n5. Resetting tracker...")
    tracker.reset()
    
    print("\n6. Getting stats after reset...")
    stats_reset = tracker.get_stats()
    print("Stats after reset:")
    for key, value in stats_reset.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_cpu_tracker()
