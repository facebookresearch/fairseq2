# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import final
import ray

from torch import Tensor
from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import SequenceBatch
from fairseq2.metrics import MetricBag
from fairseq2.metrics.common import (
    add_nll_loss_metric,
    add_seq_batch_metrics,
    update_nll_loss_metric,
    update_seq_batch_metrics,
)
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.trainer import Trainer, TrainUnit
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.gang import Gangs
from fairseq2.models.qwen import QwenConfig

from ..common import check_vocab_info
from .config import RLTrainConfig, VllmRayActorConfig, HFRayActorConfig
from .dataset import (
    PromptBatch,
    LM_RL_DATASET,
    LMRLDataset,
    LMRLDatasetConfig,
    open_rl_dataset,
)
from .grpo import GRPOTrainUnit, create_grpo_unit, GrpoUnitConfig
from .remote_model import create_vllm_remote_model, create_hf_remote_model, RemoteModel
from fairseq2.logging import log
from fairseq2.recipe.component import register_component
from .utils import register_rl_train_unit, get_parameter_converter

@final
class RLRecipe(TrainRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            LM_RL_DATASET,
            LMRLDataset,
            LMRLDatasetConfig,
            opener=open_rl_dataset,
        )
        # register_component(container, TrainUnit, "grpo", config_kls=GrpoUnitConfig, factory=create_grpo_unit)
        # register_component(container, RemoteModel, "vllm", config_kls=VllmRayActorConfig, factory=create_vllm_remote_model)
        # register_component(container, RemoteModel, "hf", config_kls=HFRayActorConfig, factory=create_hf_remote_model)
        # register_rl_train_unit(
        #     container,
        #     "grpo",
        #     GRPOTrainUnit,
        #     GrpoUnitConfig,
        #     create_grpo_unit
        # )

    def pick_ray_actor_creator(self, actor_config):
        match actor_config:
            case VllmRayActorConfig():
                return create_vllm_remote_model
            case HFRayActorConfig():
                return create_hf_remote_model
            case _:
                raise TypeError

    @override
    def create_trainer(self, context: RecipeContext) -> Trainer:
        check_vocab_info(context)

        config = context.config.as_(RLTrainConfig)

        dataset = context.default_dataset.as_(LMRLDataset)

        data_reader = dataset.create_reader(
            config.dataset.train_split,
            context.default_tokenizer,
            context.gangs,
            options=config.dataset.read_options,
        )

        context.model._convert_parameter = get_parameter_converter(context.model.config)
        
        # setting up ray actors

        if context.gangs.dp.rank == 0:
            # we only communicate with ray from 0th DP rank
            ray.init(
                runtime_env={"env_vars": {
                            "PYTHONPATH": os.environ["PYTHONPATH"],
                            "TORCHINDUCTOR_CACHE_DIR": "/scratch/vllm_cache"
                        }
                    },
                address=f"ray://{config.vllm.ray_cluster_ip_address}:10001",
                namespace="vllm_workers",
            )

        ray_actors = {}
        vocab_size = context.default_tokenizer.vocab_info.size

        # go over actor configs and initialize all of them
        for actor_config in config.vllm.ray_actors:                
            actor_creator = self.pick_ray_actor_creator(actor_config)
            if (
                isinstance(context.model.config, QwenConfig)
                and actor_config.ray_actor_name == "vllm_policy"
            ):
                # this is the hack to make sure qwen does not sample OOV tokens w.r.t the tokenizer
                # this is a known 'bug'
                actor_config.vllm_sampling_params["allowed_token_ids"] = list(
                    range(vocab_size)
                )

            log.info(f"Setting up '{actor_config.ray_actor_name}' vllm actor")
            actor = actor_creator(
                gangs=context.gangs, actor_config=actor_config
            )
            ray_actors[actor_config.ray_actor_name] = actor

        # checked non-blocked actor status
        if context.gangs.dp.rank == 0 and context.gangs.tp.rank == 0:
            model_status = []
            for actor_config in config.vllm.ray_actors:
                if not actor_config.blocking_initialization:
                    model_status.extend(
                        ray_actors[actor_config.ray_actor_name].is_model_ready()
                    )
            ray.get(model_status)

        # unit_constructor = context.resolver.resolve(TrainUnit, config.train_unit.name)
        unit_constructor = create_grpo_unit
        unit = unit_constructor(context.model, context.gangs, config.train_unit, ray_actors)

        return context.create_trainer(unit, data_reader)

    @property
    @override
    def config_kls(self) -> type[object]:
        return RLTrainConfig