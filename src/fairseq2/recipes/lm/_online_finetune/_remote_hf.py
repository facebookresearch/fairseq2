# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import ray
import re
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.nn import Module
from typing_extensions import override


from fairseq2.gang import Gangs
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes.config import get_config_section
from fairseq2.recipes.lm._online_finetune._common import (
    MyWorker,
    NoEnvLLM,
    NoEnvPipeline,
    stateless_init_process_group,
)
from fairseq2.logging import log
from transformers import (
    LlamaModel,
    LlamaPreTrainedModel,
    TextClassificationPipeline,
    AutoTokenizer,
)
from torch import nn
import torch
from typing import Dict


class RemoteHFModel:
    def __init__(
        self,
        ray_actor_name: str,
        num_replicas: int,
        tensor_parallel_size: int,
        gangs: Gangs,
    ):
        self._gangs = gangs

        self.num_replicas = num_replicas
        self.ray_actor_name = ray_actor_name

        self.hf_workers = []
        self.update_process_groups = []

        if gangs.dp.rank != 0 and gangs.tp.rank != 0:
            raise ValueError("hf worker should only be initialized on DP & TP rank 0")

        for replica_i in range(self.num_replicas):
            self.setup_replica(replica_i, tensor_parallel_size)

        self._tensor_parallel_size = tensor_parallel_size

    def setup_replica(self, replica_i: int, tensor_parallel_size):
        if (
            len(self.hf_workers) != replica_i
            or len(self.update_process_groups) != replica_i
        ):
            raise RuntimeError(
                "new replica is being created while previous ones are not setup yet"
            )

        hf_worker = self.setup_hf_worker(
            f"{self.ray_actor_name}_{replica_i}", tensor_parallel_size, self._gangs
        )
        self.hf_workers.append(hf_worker)

        log.info(f"Replica {replica_i} setup completed")

    def setup_hf_worker(self, ray_actor_name, tensor_parallel_size, gangs: Gangs):

        pg_inference = placement_group(
            [{"GPU": 1, "CPU": 0}] * tensor_parallel_size,
            strategy="STRICT_PACK",
        )

        ray.get(pg_inference.ready())

        scheduling_inference = PlacementGroupSchedulingStrategy(
            placement_group=pg_inference,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )

        llm = NoEnvPipeline.options(
            name=ray_actor_name,
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=scheduling_inference,
            get_if_exists=True,
        ).remote()

        # we block here until the engine is initialized
        ray.get(llm.is_ready.remote())

        return llm

    def rollout_from_model(self, prompt_list, sampling_params=None, string_input=False):
        raise NotImplementedError(
            "RemoteHFModel.rollout_from_model is not implemented. "
        )

    def reward_from_model(self, prompt_list, batch_size=64):
        # NOTE: need to batch inputs to hf.encode model for current models that aren't supported by hf
        rewards = []
        outputs = []
        replica_counter = 0
        for i in range(0, len(prompt_list), batch_size):
            prompt_chunk = prompt_list[i : i + batch_size]

            outputs.append(
                self.hf_workers[replica_counter % self.num_replicas].__call__.remote(
                    prompt_chunk
                )
            )
            replica_counter += 1

        ray_outputs = ray.get(outputs)

        ray_outputs_flat = [o for sublist in ray_outputs for o in sublist]

        # rewards = [o.outputs.data.item() for o in ray_outputs_flat]
        return ray_outputs_flat

    def reward_from_generative_model(self, prompt_list):

        raise NotImplementedError(
            "RemoteHFModel.reward_from_generative_model is not implemented. "
        )
