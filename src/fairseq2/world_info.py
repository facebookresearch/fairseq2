# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.utils.env import (
    Environment,
    EnvironmentVariableError,
    get_local_rank,
    get_local_world_size,
    get_rank,
    get_world_size,
)


def get_world_info() -> WorldInfo:
    return get_dependency_resolver().resolve(WorldInfo)


@dataclass(frozen=True)
class WorldInfo:
    rank: int
    size: int
    local_rank: int
    local_size: int

    @staticmethod
    def from_env(env: Environment) -> WorldInfo:
        rank = get_rank(env)

        world_size = get_world_size(env)

        if rank is not None:
            if world_size is None:
                msg = "RANK and WORLD_SIZE environment variables are expected to be specified together, but WORLD_SIZE is not specified."

                raise EnvironmentVariableError("WORLD_SIZE", msg)

            if rank >= world_size:
                msg = f"RANK environment variable is expected to be less than WORLD_SIZE ({world_size}), but is {rank} instead."

                raise EnvironmentVariableError("RANK", msg)
        elif world_size is not None:
            msg = "RANK and WORLD_SIZE environment variables are expected to be specified together, but RANK is not specified."

            raise EnvironmentVariableError("RANK", msg)
        else:
            rank = 0

            world_size = 1

        local_rank = get_local_rank(env)

        local_world_size = get_local_world_size(env)

        if local_rank is not None:
            if local_world_size is None:
                msg = "LOCAL_RANK and LOCAL_WORLD_SIZE environment variables are expected to be specified together, but LOCAL_WORLD_SIZE is not specified."

                raise EnvironmentVariableError("LOCAL_WORLD_SIZE", msg)

            if local_rank >= local_world_size:
                msg = f"LOCAL_RANK environment variable is expected to be less than LOCAL_WORLD_SIZE ({local_world_size}), but is {local_rank} instead."

                raise EnvironmentVariableError("LOCAL_RANK", msg)
        elif local_world_size is not None:
            msg = "LOCAL_RANK and LOCAL_WORLD_SIZE environment variables are expected to be specified together, but LOCAL_RANK is not specified."

            raise EnvironmentVariableError("LOCAL_RANK", msg)
        else:
            local_rank = 0

            local_world_size = 1

        return WorldInfo(rank, world_size, local_rank, local_world_size)
