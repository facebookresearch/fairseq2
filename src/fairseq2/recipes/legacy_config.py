# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.dependency import DependencyResolver
from fairseq2.recipes.config_manager import StandardConfigManager
from fairseq2.recipes.gang import GangConfig
from fairseq2.recipes.metrics import MetricRecordersConfig
from fairseq2.typing import DataClass


def _set_legacy_config(resolver: DependencyResolver, config: DataClass) -> None:
    config_dict: dict[str, object] = {}

    def set_gang_config() -> None:
        monitored_gang = getattr(config, "monitored_gang", False)

        tensor_parallel_size = getattr(config, "tensor_parallel_size", 1)

        config_dict["gang"] = GangConfig(
            monitored=monitored_gang,
            tensor_parallel_size=tensor_parallel_size,
        )

    def set_checkpoint_search_dir() -> None:
        search_dir = getattr(config, "resume_checkpoint_dir", None)
        if search_dir is None:
            search_dir = getattr(config, "checkpoint_dir", None)

        config_dict["checkpoint_search_dir"] = search_dir

    def set_score_config() -> None:
        config_dict["score_metric"] = getattr(config, "score_metric", None)

        config_dict["lower_score_better"] = getattr(config, "lower_score_better", False)

    def set_metric_recorders_config() -> None:
        wandb_project = getattr(config, "wandb_project", None)
        wandb_run_name = getattr(config, "wandb_run_name", None)

        if wandb_project is not None:
            config_dict["metric_recorders"] = MetricRecordersConfig(
                wandb=True, wandb_project=wandb_project, wandb_run_name=wandb_run_name
            )

    set_gang_config()
    set_checkpoint_search_dir()
    set_score_config()
    set_metric_recorders_config()

    config_manager = resolver.resolve(StandardConfigManager)

    config_manager.update_config_dict(config_dict)
