# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from fairseq2.dependency import DependencyContainer, DependencyResolver
from fairseq2.utils.env import get_int_from_env


def get_world_size() -> int:
    """Return the world size of the running job."""
    value = get_int_from_env("WORLD_SIZE")

    return 1 if value is None else value


def get_rank() -> int:
    """Return the rank of this process in the running job."""
    value = get_int_from_env("RANK", allow_zero=True)

    return 0 if value is None else value


def get_local_world_size() -> int:
    """Return the local world size of the running job."""
    value = get_int_from_env("LOCAL_WORLD_SIZE")

    return 1 if value is None else value


def get_local_rank() -> int:
    """Return the local rank of this process in the running job."""
    value = get_int_from_env("LOCAL_RANK", allow_zero=True)

    return 0 if value is None else value


@dataclass(frozen=True)
class RuntimeContext:
    """Holds contextual runtime information."""

    world_size: int
    rank: int
    local_world_size: int
    local_rank: int


def register_objects(container: DependencyContainer) -> None:
    container.register_factory(RuntimeContext, _create_runtime_context)


def _create_runtime_context(resolver: DependencyResolver) -> RuntimeContext:
    return RuntimeContext(
        get_world_size(), get_rank(), get_local_world_size(), get_local_rank()
    )
