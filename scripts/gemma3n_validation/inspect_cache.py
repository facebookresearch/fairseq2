#!/usr/bin/env python3
"""Check DynamicCache structure."""

from transformers.cache_utils import DynamicCache

cache = DynamicCache()
print("DynamicCache attributes:")
print(dir(cache))
print("\nDynamicCache type:", type(cache))
