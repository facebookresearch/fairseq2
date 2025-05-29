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

from vllm import SamplingParams
from vllm.engine.arg_utils import PoolerConfig
from vllm.utils import get_ip, get_open_port

from fairseq2.gang import Gangs
from fairseq2.models.sequence import SequenceBatch
from fairseq2.recipes.config import get_config_section
from fairseq2.recipes.lm._online_finetune._common import (
    MyWorker,
    NoEnvLLM,
    stateless_init_process_group,
)
from fairseq2.logging import log


class RemoteModelHandler(ABC):
    @abstractmethod
    def create(self, gangs: Gangs, unit_config: object) -> RemoteHFModel: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


@dataclass(kw_only=True)
class HFEngineArgs:
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
class HFRayActorConfig:
    ray_actor_name: str = "dummy"
    num_replicas: int = 1
    hf_engine_args: HFArgs = field(default_factory=lambda: HFArgs())
    hf_sampling_params: Dict[str, Any] = field(default_factory=lambda: {})
    init_update_process_group: bool = False


class RemoteHFModelHandler(RemoteModelHandler):
    @override
    def create(self, gangs: Gangs, actor_config: HFRayActorConfig) -> RemoteHFModel:
        if gangs.dp.rank == 0 and gangs.tp.rank == 0:
            # hf worker is only created on the first DP ranks given how many replicas we use
            remote_hf_model = RemoteHFModel(
                actor_config.ray_actor_name,
                actor_config.num_replicas,
                actor_config.hf_engine_args,
                actor_config.hf_sampling_params,
                actor_config.init_update_process_group,
                gangs,
            )
        else:
            remote_hf_model = None

        gangs.root.barrier()

        return remote_hf_model

    @property
    @override
    def name(self) -> str:
        "hf_model"

    @property
    @override
    def config_kls(self) -> type[object]:
        return HFRayActorConfig


class RemoteHFModel:
    def __init__(
        self,
        ray_actor_name: str,
        num_replicas: int,
        hf_engine_args: HFEngineArgs,
        sampling_params: dict,
        init_update_process_group: bool,
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
            self.setup_replica(replica_i, hf_engine_args, init_update_process_group)
            # gangs.root.barrier()/

        # populate sampling params using all values that were passed in the config
        self.sampling_params = SamplingParams(**sampling_params)

        self._hf_engine_args = hf_engine_args

    def setup_replica(self, replica_i: int, hf_engine_args, init_update_process_group):
        if (
            len(self.hf_workers) != replica_i
            or len(self.update_process_groups) != replica_i
        ):
            raise RuntimeError(
                "new replica is being created while previous ones are not setup yet"
            )

        hf_worker = self.setup_hf_worker(
            f"{self.ray_actor_name}_{replica_i}", hf_engine_args, self._gangs
        )
        self.hf_workers.append(hf_worker)

        update_process_group = self.setup_process_group_for_model_sync(
            replica_i, hf_engine_args.tensor_parallel_size, init_update_process_group
        )
        self.update_process_groups.append(update_process_group)

        log.info(f"Replica {replica_i} setup completed")

    def setup_hf_worker(
        self, ray_actor_name, hf_engine_args: HFEngineArgs, gangs: Gangs
    ):

        pg_inference = placement_group(
            [{"GPU": 1, "CPU": 0}] * hf_engine_args.tensor_parallel_size,
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
            model=hf_engine_args.model,
            tokenizer=hf_engine_args.tokenizer,
            enforce_eager=hf_engine_args.enforce_eager,
            worker_cls=MyWorker,
            tensor_parallel_size=hf_engine_args.tensor_parallel_size,
            task=hf_engine_args.task,
            trust_remote_code=hf_engine_args.trust_remote_code,
            model_impl=hf_engine_args.model_impl,
            hf_overrides=hf_engine_args.hf_overrides,
            override_pooler_config=hf_engine_args.override_pooler_config,
            dtype=hf_engine_args.dtype,
            distributed_executor_backend="ray",
        )

        # we block here until the engine is initialized
        ray.get(llm.is_ready.remote())

        return llm

    def setup_process_group_for_model_sync(
        self, replica_i, hf_tensor_parallel_size, init_update_process_group=True
    ):
        if not init_update_process_group:
            return None

        master_port = get_open_port()
        master_address = get_ip()

        print(f"{master_port} {master_address}")

        print("init pg on hf host")
        handle = self.hf_workers[replica_i].collective_rpc.remote(
            "init_weight_update_group",
            args=(master_address, master_port, 1, hf_tensor_parallel_size + 1),
        )

        print("init pg on train host")
        model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            0,
            hf_tensor_parallel_size + 1,
            self._gangs.dp.device,
        )
        ray.get(handle)

        return model_update_group

    def sync_weights_with_hf(self, train_model):
        """
        trainer_process_group must connect training process with hf_model processes
        """

        # iterate over all replicas
        for replica_i in range(self.num_replicas):
            for name, p in train_model.module.named_parameters():
                name = name.replace("._checkpoint_wrapped_module", "")
                # print(f'sync call {name}')
                handle = self.hf_workers[replica_i].collective_rpc.remote(
                    "update_weight", args=(name, p.dtype, p.shape)
                )
                self.update_process_groups[replica_i].broadcast(
                    p, src=0, stream=torch.cuda.current_stream()
                )
                ray.get(handle)

    def rollout_from_model(self, prompt_list, sampling_params=None, string_input=False):
        if sampling_params is None:
            sampling_params = self.sampling_params

        prompt_argname = "prompts" if string_input else "prompt_token_ids"

        # Split prompts evenly across replicas
        chunk_size = len(prompt_list) // self.num_replicas
        remainder = len(prompt_list) % self.num_replicas

        chunks = []
        start = 0
        for i in range(self.num_replicas):
            # Add one extra item to some chunks if division isn't even
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(prompt_list[start:end])
            start = end

        # Process each chunk with a different replica
        outputs = []
        for replica_i, chunk in enumerate(chunks):
            if len(chunk) > 0:  # Only send non-empty chunks
                generate_args = {
                    prompt_argname: chunk,
                    "sampling_params": sampling_params,
                    "use_tqdm": False,
                }
                output = self.hf_workers[replica_i].generate.remote(**generate_args)
                outputs.append(output)

        # block till generation is done
        results = ray.get(outputs)

        rollouts = []
        for chunk_result in results:
            rollouts.extend(chunk_result)

        return rollouts

    def reward_from_model(self, prompt_list, batch_size=64):
        # NOTE: need to batch inputs to hf.encode model for current models that aren't supported by hf
        rewards = []
        outputs = []
        replica_counter = 0
        for i in range(0, len(prompt_list), batch_size):
            prompt_chunk = prompt_list[i : i + batch_size]
            outputs.append(
                self.hf_workers[replica_counter % self.num_replicas].encode.remote(
                    prompt_chunk,
                    use_tqdm=False,
                )
            )
            replica_counter += 1
        ray_outputs = ray.get(outputs)
        ray_outputs_flat = [o for sublist in ray_outputs for o in sublist]
        rewards = [o.outputs.data.item() for o in ray_outputs_flat]
        return rewards

    def reward_from_generative_model(self, prompt_list):

        def extract_score(output):
            matches = re.findall(
                r"<score>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score>", output
            )
            return float(matches[-1]) if matches else 0.0

        rewards = []
        rollouts = self.rollout_from_model(prompt_list=prompt_list, string_input=True)
        rewards = [extract_score(o.outputs[0].text) for o in rollouts]

        return rewards
