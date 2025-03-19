# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.cli import CliArgumentError, CliCommandError
from fairseq2.cluster import (
    ClusterError,
    ClusterHandler,
    ClusterResolver,
    UnknownClusterError,
)
from fairseq2.context import RuntimeContext


def set_torch_distributed_variables(context: RuntimeContext, cluster: str) -> None:
    cluster_handlers = context.get_registry(ClusterHandler)

    cluster_resolver = ClusterResolver(cluster_handlers, context.env)

    try:
        handler = cluster_resolver.get(cluster)
    except UnknownClusterError as ex:
        s = ", ".join(ex.supported_clusters)

        raise CliArgumentError(
            "cluster", f"'{ex.cluster}' is not a known cluster. Must be one of: auto, none, {s}"  # fmt: skip
        ) from None

    try:
        handler.set_torch_distributed_variables()
    except ClusterError as ex:
        if ex.cluster == "slurm":
            message = f"'{ex.cluster}' cluster environment cannot be set. See the logged stack trace for details. If you are within an allocated Slurm job (i.e. `salloc`), make sure to run with `srun`. If you want to run without Slurm, use `--cluster none`."
        else:
            message = f"'{ex.cluster}' cluster environment cannot be set. See the logged stack trace for details."

        raise CliCommandError(message) from ex
