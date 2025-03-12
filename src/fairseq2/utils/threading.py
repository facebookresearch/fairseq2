# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ParamSpec, Protocol, TypeVar, final

from typing_extensions import override

from fairseq2.utils.env import InvalidEnvironmentVariableError, get_local_world_size

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
class StandardThreadPool(ThreadPool):
    _executor: ThreadPoolExecutor

    def __init__(self, max_num_workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_num_workers)

    @override
    def queue(
        self, callback: Action[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]:
        return self._executor.submit(callback, *args, **kwargs)


_default_pool: ThreadPool | None = None


def get_default_thread_pool() -> ThreadPool:
    global _default_pool

    if _default_pool is None:
        num_threads = get_num_threads(os.environ)

        # Due to GIL, threads in CPython are typically used to overlap I/O with
        # CPU work; therefore, we can slightly oversubscribe here (i.e. +4).
        _default_pool = StandardThreadPool(max_num_workers=num_threads + 4)

    return _default_pool


def get_num_threads(env: Mapping[str, str]) -> int:
    try:
        num_procs = get_local_world_size(env)
    except InvalidEnvironmentVariableError as ex:
        raise ThreadingError(
            "The local world size cannot be determined from the environment variables. See the nested exception for details."
        ) from ex

    # To prevent thread oversubscription, we distribute cores evenly across the
    # gang processes.
    return _get_num_cpus(num_procs)


def _get_num_cpus(num_procs: int) -> int:
    num_cpus = os.cpu_count()

    affinity_mask = os.sched_getaffinity(0)

    if num_cpus is None or affinity_mask is None:
        raise ThreadingError(
            "The number of CPUs of the host machine cannot be determined."
        )

    # We should not exceed the number of cores available in the affinity mask.
    return min(max(num_cpus // num_procs, 1), len(affinity_mask))


class ThreadingError(Exception):
    pass
