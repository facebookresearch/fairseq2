# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ParamSpec, Protocol, TypeVar, final

from typing_extensions import override

T = TypeVar("T")


class ThreadLocalStorage(ABC):
    @abstractmethod
    def get(self, key: str, default_factory: Callable[[], T]) -> T: ...


@final
class _StandardThreadLocalStorage(ThreadLocalStorage):
    def __init__(self) -> None:
        self._local = threading.local()

    @override
    def get(self, key: str, default_factory: Callable[[], T]) -> T:
        value = getattr(self._local, key, None)
        if value is None:
            value = default_factory()

            setattr(self._local, key, value)

        return value


P = ParamSpec("P")

R_co = TypeVar("R_co", covariant=True)


class Action(Protocol[P, R_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...


R = TypeVar("R")


class ThreadPool(ABC):
    @abstractmethod
    def queue(
        self, callback: Action[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]: ...


@final
class _StandardThreadPool(ThreadPool):
    @staticmethod
    def create_default(local_world_size: int) -> _StandardThreadPool:
        num_threads = get_num_threads(local_world_size)

        # Due to GIL, threads in CPython are typically used to overlap I/O with
        # CPU work; therefore, we can slightly oversubscribe here (i.e. +4).
        return _StandardThreadPool(max_num_workers=num_threads + 4)

    def __init__(self, max_num_workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_num_workers)

    @override
    def queue(
        self, callback: Action[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]:
        return self._executor.submit(callback, *args, **kwargs)


def get_num_threads(local_world_size: int) -> int:
    num_cpus = os.cpu_count()

    affinity_mask = os.sched_getaffinity(0)

    if num_cpus is None or affinity_mask is None:
        raise RuntimeError("Number of CPUs on the host machine cannot be determined.")

    # We should not exceed the number of cores available in the affinity mask.
    return min(max(num_cpus // local_world_size, 1), len(affinity_mask))
