# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

from fairseq2.context import RuntimeContext
from fairseq2.recipes.cluster import ClusterHandler, ClusterResolver


def set_torch_distributed_variables(context: RuntimeContext, cluster: str) -> None:
    cluster_handlers = context.get_registry(ClusterHandler)

    cluster_resolver = ClusterResolver(cluster_handlers, os.environ)

    handler = cluster_resolver.get(cluster)

    handler.set_torch_distributed_variables()
