# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.cluster import ClusterHandler, RayClusterHandler, SlurmClusterHandler
from fairseq2.context import RuntimeContext


def register_clusters(context: RuntimeContext) -> None:
    registry = context.get_registry(ClusterHandler)

    # Slurm
    handler = SlurmClusterHandler(context.env)

    registry.register(handler.supported_cluster, handler)

    # Ray
    ray_handler = RayClusterHandler(context.env)
    registry.register(ray_handler.supported_cluster, ray_handler)
