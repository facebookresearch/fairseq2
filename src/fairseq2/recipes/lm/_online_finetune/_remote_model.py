# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Union
from typing_extensions import override
from vllm.engine.arg_utils import PoolerConfig
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipes.lm._online_finetune._remote_hf import RemoteHFModel
from fairseq2.recipes.lm._online_finetune._remote_vllm import RemoteVllmModel
from fairseq2.utils.structured import StructureError, structure


class RemoteModelHandler(ABC):
    @abstractmethod
    def create(self, gangs: Gangs, unit_config: object) -> RemoteVllmModel: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


@dataclass(kw_only=True)
class VllmEngineArgs:
    model: str = "/checkpoint/ram/kulikov/gsm8k_8b_sft/checkpoints/step_20"
    tokenizer: str = "/datasets/pretrained-llms/Llama-3.1-8B-Instruct"
    task: str = "generate"
    tensor_parallel_size: int = 4
    trust_remote_code: bool = False
    model_impl: str = "auto"
    enforce_eager: bool = True
    hf_overrides: object = None
    dtype: str = "auto"
    override_pooler_config: PoolerConfig = field(default_factory=lambda: PoolerConfig())


@dataclass(kw_only=True)
class RayActorConfig(ABC):
    ray_actor_name: str = "dummy"
    backend: str = "vllm"  # vllm or hf
    num_replicas: int = 1
    init_update_process_group: bool = False


@dataclass(kw_only=True)
class VllmRayActorConfig(RayActorConfig):
    vllm_engine_args: VllmEngineArgs = field(default_factory=lambda: VllmEngineArgs())
    vllm_sampling_params: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass(kw_only=True)
class HFRayActorConfig(RayActorConfig):
    tensor_parallel_size: int = 4


class RemoteRayModelHandler(RemoteModelHandler):
    @override
    def create(
        self, gangs: Gangs, actor_config: RayActorConfig
    ) -> Union[RemoteVllmModel, RemoteHFModel]:
        if gangs.dp.rank == 0 and gangs.tp.rank == 0:
            # vllm worker is only created on the first DP ranks given how many replicas we use
            if actor_config.backend == "vllm":
                actor_config = structure(actor_config, VllmRayActorConfig)
                remote_vllm_model = RemoteVllmModel(
                    actor_config.ray_actor_name,
                    actor_config.num_replicas,
                    actor_config.vllm_engine_args,
                    actor_config.vllm_sampling_params,
                    actor_config.init_update_process_group,
                    gangs,
                )
            else:
                actor_config = structure(actor_config, HFRayActorConfig)
                remote_vllm_model = RemoteHFModel(
                    actor_config.ray_actor_name,
                    actor_config.num_replicas,
                    actor_config.tensor_parallel_size,
                    gangs,
                )
        else:
            remote_vllm_model = None

        gangs.root.barrier()

        return remote_vllm_model

    @property
    @override
    def name(self) -> str:
        "vllm_model"

    @property
    @override
    def config_kls(self) -> type[object]:
        return RayActorConfig
