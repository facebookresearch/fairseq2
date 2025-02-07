# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

from fairseq2.context import RuntimeContext
from fairseq2.recipes.cluster import ClusterHandler, SlurmClusterHandler


def _register_clusters(context: RuntimeContext) -> None:
    registry = context.get_registry(ClusterHandler)

    registry.register("slurm", SlurmClusterHandler(os.environ))
