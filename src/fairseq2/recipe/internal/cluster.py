# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from fairseq2.cluster import _ClusterResolver
from fairseq2.error import OperationalError, InvalidOperationError
from fairseq2.logging import log
from fairseq2.recipe.config import CommonSection
from fairseq2.recipe.error import ConfigError


@final
class _ClusterPreparer:
    def __init__(
        self, section: CommonSection, cluster_resolver: _ClusterResolver
    ) -> None:
        self._section = section
        self._cluster_resolver = cluster_resolver

    def prepare(self) -> None:
        try:
            handler = self._cluster_resolver.resolve(self._section.cluster)
        except LookupError:
            s = ", ".join(self._cluster_resolver.supported_clusters)

            raise ConfigError(
                f"{self._section.cluster} is not a known cluster. `common.cluster` must be one of auto, none, {s}."
            ) from None

        try:
            handler.set_torch_distributed_env_variables()
        except InvalidOperationError:
            if handler.cluster == "slurm":
                message = f"{handler.cluster} cluster not detected. If you are within an allocated job (i.e. salloc), make sure to run with srun. If you want to run locally (e.g. via torchrun), use `--config common.cluster=none`."
            else:
                message = f"{handler.cluster} cluster not detected."

            raise ConfigError(message) from None
        except RuntimeError as ex:
            raise OperationalError(
                f"torch.distributed environment variables cannot be set from the {handler.cluster} cluster settings."
            ) from ex

        if self._section.cluster == "auto":
            if handler.cluster == "none":
                log.info("No cluster detected.")
            else:
                log.info("torch.distributed environment variables set for the {} cluster.", handler.cluster)  # fmt: skip
