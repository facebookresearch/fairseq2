# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import Namespace

from fairseq2.cli import CliArgumentError, CliCommandError
from fairseq2.cluster import (
    ClusterError,
    ClusterHandler,
    ClusterResolver,
    UnknownClusterError,
)
from fairseq2.dependency import DependencyResolver
from fairseq2.utils.env import get_env


def set_torch_distributed_variables(resolver: DependencyResolver) -> None:
    env = get_env(resolver)

    handlers = resolver.resolve_all(ClusterHandler)

    cluster_resolver = ClusterResolver(handlers, env)

    args = resolver.resolve(Namespace)

    try:
        handler = cluster_resolver.get(args.cluster)
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
