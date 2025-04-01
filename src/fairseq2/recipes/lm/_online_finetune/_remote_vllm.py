# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict
from typing_extensions import override
import torch

from torch.nn import Module

from fairseq2.models.sequence import SequenceBatch
from fairseq2.gang import Gangs
from fairseq2.recipes.lm._online_finetune._common import (
    NoEnvLLM,
    MyWorker,
    stateless_init_process_group,
)
from fairseq2.recipes.config import (
    get_config_section,
)
from vllm import SamplingParams
from ray.util.placement_group import placement_group
from vllm.utils import get_ip, get_open_port
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray


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
    tensor_parallel_size: int = 4
    enforce_eager: bool = True


@dataclass(kw_only=True)
class VllmSamplingParams:
    n: int = 4
    temperature: float = 1.0
    max_tokens: int = 1024
    prompt_logprobs: int | None = None
    detokenize: bool = True


@dataclass(kw_only=True)
class VllmRayActorConfig:
    ray_actor_name: str = "dummy"
    vllm_engine_args: VllmEngineArgs = field(default_factory=lambda: VllmEngineArgs())
    vllm_sampling_params: VllmSamplingParams = field(
        default_factory=lambda: VllmSamplingParams()
    )
    init_update_process_group: bool = False


@dataclass(kw_only=True)
class VllmConfig:
    ray_cluster_ip_address: str = "dummy"
    ray_actor_name: str = "dummy"
    vllm_engine_args: VllmEngineArgs = field(default_factory=lambda: VllmEngineArgs())
    vllm_sampling_params: VllmSamplingParams = field(
        default_factory=lambda: VllmSamplingParams()
    )
    init_update_process_group: bool = False


class RemoteVllmModelHandler(RemoteModelHandler):
    @override
    def create(self, gangs: Gangs, actor_config: VllmRayActorConfig) -> RemoteVllmModel:
        if gangs.dp.rank == 0 and gangs.tp.rank == 0:
            # vllm worker is only created on the first DP rank (incuding all TP ranks)
            remote_vllm_model = RemoteVllmModel(
                actor_config.ray_actor_name,
                actor_config.vllm_engine_args,
                actor_config.vllm_sampling_params,
                actor_config.init_update_process_group,
                gangs,
            )
        else:
            remote_vllm_model = None

        return remote_vllm_model

    @property
    @override
    def name(self) -> str:
        "vllm_model"

    @property
    @override
    def config_kls(self) -> type[object]:
        return VllmConfig


class RemoteVllmModel:
    def __init__(
        self,
        ray_actor_name: str,
        vllm_engine_args: VllmEngineArgs,
        sampling_params: VllmSamplingParams,
        init_update_process_group: bool,
        gangs: Gangs,
    ):
        if gangs.dp.rank != 0 and gangs.tp.rank != 0:
            raise ValueError("vllm worker should only be initialized on DP & TP rank 0")

        # connection is done on recipe level and at this point we are connected
        # ray.init(address=f"ray://{ray_cluster_ip_address}:10001", namespace="vllm_workers")

        self._gangs = gangs
        self.vllm_model = self.setup_vllm_worker(
            ray_actor_name, vllm_engine_args, gangs
        )
        self.sampling_params = SamplingParams(
            n=sampling_params.n,
            temperature=sampling_params.temperature,
            max_tokens=sampling_params.max_tokens,
            detokenize=sampling_params.detokenize,
            prompt_logprobs=sampling_params.prompt_logprobs,
        )

        if init_update_process_group:
            self.update_process_group = self.setup_process_group_for_model_sync(
                vllm_engine_args.tensor_parallel_size
            )
        else:
            self.update_process_group = None

        self._vllm_engine_args = vllm_engine_args

    def setup_vllm_worker(
        self, ray_actor_name, vllm_engine_args: VllmEngineArgs, gangs: Gangs
    ):

        pg_inference = placement_group(
            [{"GPU": 1, "CPU": 0}] * vllm_engine_args.tensor_parallel_size,
            strategy="STRICT_PACK",
        )

        ray.get(pg_inference.ready())

        scheduling_inference = PlacementGroupSchedulingStrategy(
            placement_group=pg_inference,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )

        llm = NoEnvLLM.options(
            name=ray_actor_name,
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=scheduling_inference,
            get_if_exists=True,
        ).remote(
            model=vllm_engine_args.model,
            tokenizer=vllm_engine_args.tokenizer,
            enforce_eager=vllm_engine_args.enforce_eager,
            worker_cls=MyWorker,
            tensor_parallel_size=vllm_engine_args.tensor_parallel_size,
            distributed_executor_backend="ray",
        )

        # we block here until the engine is initialized
        ray.get(llm.is_ready.remote())

        return llm

    def setup_process_group_for_model_sync(self, vllm_tensor_parallel_size):
        master_port = get_open_port()
        master_address = get_ip()

        print(f"{master_port} {master_address}")

        print("init pg on vllm host")
        handle = self.vllm_model.collective_rpc.remote(
            "init_weight_update_group",
            args=(master_address, master_port, 1, vllm_tensor_parallel_size + 1),
        )

        print("init pg on train host")
        model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            0,
            vllm_tensor_parallel_size + 1,
            self._gangs.dp.device,
        )
        ray.get(handle)

        return model_update_group

    def sync_weights_with_vllm(self, train_model):
        """
        trainer_process_group must connect training process with vllm_model processes
        """
        for name, p in train_model.module.named_parameters():
            name = name.replace("._checkpoint_wrapped_module", "")
            # print(f'sync call {name}')
            handle = self.vllm_model.collective_rpc.remote(
                "update_weight", args=(name, p.dtype, p.shape)
            )
            self.update_process_group.broadcast(
                p, src=0, stream=torch.cuda.current_stream()
            )
            ray.get(handle)

    def rollout_from_model(self, prompt_list, sampling_params=None):
        if sampling_params is None:
            sampling_params = self.sampling_params

        outputs = ray.get(
            self.vllm_model.generate.remote(
                prompt_token_ids=prompt_list,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        )
        return outputs
