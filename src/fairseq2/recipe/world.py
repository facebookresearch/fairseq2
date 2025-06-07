# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.cluster import ClusterResolver
from fairseq2.logging import log
from fairseq2.recipe.config import CommonSection
from fairseq2.utils.env import Environment
from fairseq2.world_info import WorldInfo


@final
class WorldPreparer:
    def __init__(
        self,
        section: CommonSection,
        env: Environment,
        cluster_resolver: ClusterResolver,
    ) -> None:
        self._env = env
        self._section = section
        self._cluster_resolver = cluster_resolver

    def prepare(self) -> WorldInfo:
        handler = self._cluster_resolver.resolve(self._section.cluster)

        handler.set_torch_distributed_env_variables()

        if self._section.cluster == "auto":
            log.info("torch.distributed environment variables set for the {} cluster.", handler.supported_cluster)  # fmt: skip

        return WorldInfo.from_env(self._env)
