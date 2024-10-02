# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.dependency import DependencyResolver
from fairseq2.recipes.config_manager import StandardConfigManager
from fairseq2.recipes.gang import GangConfig
from fairseq2.typing import DataClass


def _set_legacy_config(resolver: DependencyResolver, config: DataClass) -> None:
    monitored_gang = getattr(config, "monitored_gang", False)

    tensor_parallel_size = getattr(config, "tensor_parallel_size", 1)

    config_dict: dict[str, object] = {
        "gang": GangConfig(
            monitored=monitored_gang,
            tensor_parallel_size=tensor_parallel_size,
        ),
    }

    config_manager = resolver.resolve(StandardConfigManager)

    config_manager.override_config(config_dict)
