# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

from fairseq2.cluster import ClusterHandler, ClusterResolver, SlurmHandler
from fairseq2.recipe.config import CommonSection, get_config_section
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.env import (
    get_env,
    get_local_rank,
    get_local_world_size,
    get_rank,
    get_world_size,
)


@dataclass
class WorldInfo:
    rank: int
    size: int
    local_rank: int
    local_size: int


def _create_world_info(resolver: DependencyResolver) -> WorldInfo:
    cluster_handler = _create_cluster_handler(resolver)

    env = get_env(resolver)

    cluster_handler.set_torch_distributed_env_variables()

    rank = get_rank(env)

    world_size = get_world_size(env)

    local_rank = get_local_rank(env)

    local_world_size = get_local_world_size(env)

    return WorldInfo(rank, world_size, local_rank, local_world_size)


def _create_cluster_handler(resolver: DependencyResolver) -> ClusterHandler:
    common_section = get_config_section(resolver, "common", CommonSection)

    other_handlers = resolver.resolve_all(ClusterHandler, key="alt")

    env = get_env(resolver)

    handlers: list[ClusterHandler] = [SlurmHandler(env)]

    it = chain(handlers, other_handlers)

    cluster_resolver = ClusterResolver(it, env)

    return cluster_resolver.resolve(common_section.cluster)
