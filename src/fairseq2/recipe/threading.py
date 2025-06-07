# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.cluster import WorldInfo
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.threading import FuturesThreadPool, ThreadPool, get_num_threads


def _create_thread_pool(resolver: DependencyResolver) -> ThreadPool:
    world_info = resolver.resolve(WorldInfo)

    num_threads = get_num_threads(world_info.local_size)

    # Due to GIL, threads in CPython are typically used to overlap I/O with CPU
    # work; therefore, we can slightly oversubscribe here (i.e. +4).
    return FuturesThreadPool(max_num_workers=num_threads + 4)
