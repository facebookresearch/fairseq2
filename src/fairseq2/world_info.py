# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from fairseq2.utils.env import (
    Environment,
    EnvironmentVariableError,
    maybe_get_local_rank,
    maybe_get_local_world_size,
    maybe_get_rank,
    maybe_get_world_size,
)


@dataclass(frozen=True)
class WorldInfo:
    rank: int
    size: int
    local_rank: int
    local_size: int

    @staticmethod
    def from_env(env: Environment) -> WorldInfo:
        """
        :raises EnvironmentVariableError:
        """
        rank = maybe_get_rank(env)

        world_size = maybe_get_world_size(env)

        if rank is not None:
            if world_size is None:
                raise EnvironmentVariableError(
                    "WORLD_SIZE", "RANK and WORLD_SIZE environment variables are expected to be defined together, but WORLD_SIZE is not defined."  # fmt: skip
                )

            if rank >= world_size:
                raise EnvironmentVariableError(
                    "RANK", f"RANK environment variable is expected to be less than WORLD_SIZE ({world_size}), but is {rank} instead."  # fmt: skip
                )
        elif world_size is not None:
            raise EnvironmentVariableError(
                "RANK", "RANK and WORLD_SIZE environment variables are expected to be defined together, but RANK is not defined."  # fmt: skip
            )
        else:
            rank = 0

            world_size = 1

        local_rank = maybe_get_local_rank(env)

        local_world_size = maybe_get_local_world_size(env)

        if local_rank is not None:
            if local_world_size is None:
                raise EnvironmentVariableError(
                    "LOCAL_WORLD_SIZE", "LOCAL_RANK and LOCAL_WORLD_SIZE environment variables are expected to be defined together, but LOCAL_WORLD_SIZE is not defined."  # fmt: skip
                )

            if local_rank >= local_world_size:
                raise EnvironmentVariableError(
                    "LOCAL_RANK", f"LOCAL_RANK environment variable is expected to be less than LOCAL_WORLD_SIZE ({local_world_size}), but is {local_rank} instead."  # fmt: skip
                )
        elif local_world_size is not None:
            raise EnvironmentVariableError(
                "LOCAL_RANK", "LOCAL_RANK and LOCAL_WORLD_SIZE environment variables are expected to be defined together, but LOCAL_RANK is not defined."  # fmt: skip
            )
        else:
            local_rank = 0

            local_world_size = 1

        return WorldInfo(rank, world_size, local_rank, local_world_size)
