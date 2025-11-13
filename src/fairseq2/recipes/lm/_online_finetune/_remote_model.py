# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Union
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing_extensions import override
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import PoolerConfig
from vllm.inputs import TokensPrompt, TextPrompt
from vllm.utils import get_ip, get_open_port

from fairseq2.context import RuntimeContext
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.nn._batch_layout import BatchLayout
from fairseq2.recipes.lm._online_finetune.third_party.athene import AtheneRewardPipeline
from fairseq2.recipes.lm._online_finetune.third_party.general_verifier import (
    GeneralVerifierPipeline,
)
from fairseq2.utils.structured import StructureError, structure


@dataclass(kw_only=True)
class RayActorConfig(ABC):
    ray_actor_name: str = "dummy"
    backend: str = "vllm"  # vllm or hf
    num_replicas: int = 1
    init_update_process_group: bool = False
    blocking_initialization: bool = False


@dataclass(kw_only=True)
class VllmEngineArgs:
    model: str = "/checkpoint/ram/kulikov/gsm8k_8b_sft/checkpoints/step_20"
    tokenizer: str = "/datasets/pretrained-llms/Llama-3.1-8B-Instruct"
    task: str = "generate"
    tensor_parallel_size: int = 4
    trust_remote_code: bool = False
    model_impl: str = "auto"
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int | None = None
    enable_chunked_prefill: bool = False
    hf_overrides: object = None
    dtype: str = "auto"
    override_pooler_config: PoolerConfig = field(default_factory=lambda: PoolerConfig())


@dataclass(kw_only=True)
class VllmRayActorConfig(RayActorConfig):
    vllm_engine_args: VllmEngineArgs = field(default_factory=lambda: VllmEngineArgs())
    vllm_sampling_params: Dict[str, Any] = field(default_factory=lambda: {})


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


@ray.remote
class NoEnvLLM(LLM):
    def __init__(self, *args, **kwargs):
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        del os.environ["CUDA_VISIBLE_DEVICES"]
        # os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        super().__init__(*args, **kwargs)

        self.ready = True  # Set a flag or return a signal

    def is_ready(self):
        return self.ready


@ray.remote
class NoEnvAtheneRewardPipeline(AtheneRewardPipeline):
    def __init__(self, *args, **kwargs):
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        del os.environ["CUDA_VISIBLE_DEVICES"]
        super().__init__(*args, **kwargs)
        self.ready = True  # Set a flag or return a signal

    def is_ready(self):
        return self.ready

    @property
    def name(self):
        return "athene_reward_pipeline"


@ray.remote
class NoEnvGeneralVerifierPipeline(GeneralVerifierPipeline):
    """
    This is for running general verifier pipeline with HF backend.
    It's not necessary since we can run this RM with VLLM backend,
    but it provides a good example of how to create a pipeline.
    """

    def __init__(self, *args, **kwargs):
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        del os.environ["CUDA_VISIBLE_DEVICES"]
        super().__init__(*args, **kwargs)
        self.ready = True  # Set a flag or return a signal

    def is_ready(self):
        return self.ready

    @property
    def name(self):
        return "general_verifier_pipeline"


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        print(f"vllm own rank: {rank}")
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight


def should_sync(
    sync_every_n_steps: int,
    trainer_step_nr: int | None,
    remote_model_step: int | None,
    force_sync: bool = False,
):
    if force_sync:
        return True

    if sync_every_n_steps < 0:
        return False

    if remote_model_step == trainer_step_nr:
        # we are in the middle of micro-batching / grad accumulation, no sync
        return False

    if trainer_step_nr < remote_model_step:
        raise RuntimeError(f"trainer step can not be less than remote model step")

    if trainer_step_nr % sync_every_n_steps == 0:
        return True


def maybe_sync_model(
    gangs: Gangs,
    model,
    remote_model: RemoteVllmModel | None,
    trainer_step_nr: int,
    sync_every_n_steps: int,
    force_sync: bool = False,
):
    if gangs.dp.rank == 0:
        _should_sync = should_sync(
            sync_every_n_steps, trainer_step_nr, remote_model.last_sync_step, force_sync
        )
        broadcast_list = [_should_sync]
    else:
        broadcast_list = [None]
    gangs.root.barrier()

    gangs.root.broadcast_objects(broadcast_list, source_rank=0)
    gangs.root.barrier()
    _should_sync = broadcast_list[0]

    if _should_sync:
        with model.summon_full_parameters():
            if gangs.dp.rank == 0:
                remote_model.sync_weights_with_vllm(
                    model=model,
                    trainer_step_nr=trainer_step_nr,
                    converter=model._convert_parameter,
                )
            gangs.root.barrier()


class RemoteVllmModel:
    def __init__(
        self,
        ray_actor_name: str,
        num_replicas: int,
        vllm_engine_args,
        sampling_params: dict,
        init_update_process_group: bool,
        blocking_initialization: bool,
        context: RuntimeContext,
        gangs: Gangs,
    ):
        self._gangs = gangs
        self._context = context
        self.blocking_initialization = blocking_initialization

        self.num_replicas = num_replicas
        self.ray_actor_name = ray_actor_name
        self.last_sync_step = -1

        self.vllm_workers = []
        self.update_process_groups = []
        self.tensor_parallel_size = vllm_engine_args.tensor_parallel_size
        self.init_update_process_group = init_update_process_group

        if gangs.dp.rank != 0 and gangs.tp.rank != 0:
            raise ValueError("vllm worker should only be initialized on DP & TP rank 0")

        for replica_i in range(self.num_replicas):
            vllm_worker = self.setup_vllm_worker(
                f"{self.ray_actor_name}_{replica_i}", vllm_engine_args, self._gangs
            )
            self.vllm_workers.append(vllm_worker)

        if self.blocking_initialization:
            self.setup_process_groups_for_model_sync(
                vllm_engine_args.tensor_parallel_size, init_update_process_group
            )
            log.info(f"Replica {replica_i} setup completed")
        else:
            log.info(f"Replica {replica_i} setup started")

        # populate sampling params using all values that were passed in the config
        self.sampling_params = SamplingParams(**sampling_params)

        self._vllm_engine_args = vllm_engine_args

    def is_model_ready(self):
        self.setup_process_groups_for_model_sync(
            self.tensor_parallel_size, self.init_update_process_group
        )
        ready_refs = [llm.is_ready.remote() for llm in self.vllm_workers]
        return ready_refs

    def setup_vllm_worker(self, ray_actor_name, vllm_engine_args, gangs: Gangs):

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
            worker_extension_cls="fairseq2.recipes.lm._online_finetune._remote_model.WorkerExtension",
            tensor_parallel_size=vllm_engine_args.tensor_parallel_size,
            task=vllm_engine_args.task,
            trust_remote_code=vllm_engine_args.trust_remote_code,
            model_impl=vllm_engine_args.model_impl,
            hf_overrides=vllm_engine_args.hf_overrides,
            gpu_memory_utilization=vllm_engine_args.gpu_memory_utilization,
            max_num_batched_tokens=vllm_engine_args.max_num_batched_tokens,
            enable_chunked_prefill=vllm_engine_args.enable_chunked_prefill,
            override_pooler_config=vllm_engine_args.override_pooler_config,
            dtype=vllm_engine_args.dtype,
            distributed_executor_backend="ray",
        )

        if self.blocking_initialization:
            # we block here until the engine is initialized
            ray.get(llm.is_ready.remote())

        return llm

    def setup_process_groups_for_model_sync(
        self, vllm_tensor_parallel_size, init_update_process_group=True
    ):
        for replica_i in range(self.num_replicas):
            self.setup_process_group_for_model_sync(
                replica_i, vllm_tensor_parallel_size, init_update_process_group
            )

    def setup_process_group_for_model_sync(
        self, replica_i, vllm_tensor_parallel_size, init_update_process_group=True
    ):
        if not init_update_process_group:
            return None

        master_port = get_open_port()
        master_address = get_ip()

        print(f"{master_port} {master_address}")

        print("init pg on vllm host")
        handle = self.vllm_workers[replica_i].collective_rpc.remote(
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

        self.update_process_groups.append(model_update_group)

    def sync_weights_with_vllm(self, model, trainer_step_nr, converter=None):
        """
        trainer_process_group must connect training process with vllm_model processes
        """

        # iterate over all replicas
        for replica_i in range(self.num_replicas):
            for name, p in model.module.named_parameters():
                name = name.replace(
                    "._checkpoint_wrapped_module", ""
                )  # remove fsdp added substring
                if converter:
                    name, p = converter(name, p, model.config)
                handle = self.vllm_workers[replica_i].collective_rpc.remote(
                    "update_weight", args=(name, p.dtype, p.shape)
                )
                self.update_process_groups[replica_i].broadcast(
                    p, src=0, stream=torch.cuda.current_stream()
                )
                ray.get(handle)
        self.last_sync_step = trainer_step_nr

    def rollout_from_model(self, prompt_list, sampling_params=None, string_input=False):
        if sampling_params is None:
            sampling_params = self.sampling_params

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
                    "prompts": [
                        (
                            TextPrompt(prompt=item)
                            if string_input
                            else TokensPrompt(prompt_token_ids=item)
                        )
                        for item in chunk
                    ],
                    "sampling_params": sampling_params,
                    "use_tqdm": False,
                }
                output = self.vllm_workers[replica_i].generate.remote(**generate_args)
                outputs.append(output)

        # block till generation is done
        results = ray.get(outputs)

        rollouts = []
        for chunk_result in results:
            rollouts.extend(chunk_result)

        return rollouts

    def reward_from_model(self, prompt_list, batch_size=64):
        # NOTE: need to batch inputs to vllm.encode model for current models that aren't supported by vllm
        rewards = []
        outputs = []
        replica_counter = 0
        for i in range(0, len(prompt_list), batch_size):
            prompt_chunk = prompt_list[i : i + batch_size]
            outputs.append(
                self.vllm_workers[replica_counter % self.num_replicas].encode.remote(
                    prompt_chunk,
                    use_tqdm=False,
                )
            )
            replica_counter += 1
        ray_outputs = ray.get(outputs)
        ray_outputs_flat = [o for sublist in ray_outputs for o in sublist]
        rewards = [o.outputs.data.item() for o in ray_outputs_flat]

        return rewards


@dataclass(kw_only=True)
class HFRayActorConfig(RayActorConfig):
    pipeline_name: str = ""
    tensor_parallel_size: int = 4
    blocking_initialization: bool = False


class RemoteHFModel:
    def __init__(
        self,
        ray_actor_name: str,
        num_replicas: int,
        tensor_parallel_size: int,
        pipeline_name: str,
        context: RuntimeContext,
        gangs: Gangs,
        blocking_initialization: bool = False,
    ):
        self._gangs = gangs

        self.num_replicas = num_replicas
        self.ray_actor_name = ray_actor_name
        self.blocking_initialization = blocking_initialization
        self.hf_workers = []

        if gangs.dp.rank != 0 and gangs.tp.rank != 0:
            raise ValueError("hf worker should only be initialized on DP & TP rank 0")

        for replica_i in range(self.num_replicas):
            self.setup_replica(replica_i, tensor_parallel_size, pipeline_name, context)

        self._tensor_parallel_size = tensor_parallel_size

    def setup_replica(
        self,
        replica_i: int,
        tensor_parallel_size: int,
        pipeline_name: str,
        context: RuntimeContext,
    ):
        if len(self.hf_workers) != replica_i:
            raise RuntimeError(
                "new replica is being created while previous ones are not setup yet"
            )

        hf_worker = self.setup_hf_worker(
            f"{self.ray_actor_name}_{replica_i}",
            tensor_parallel_size,
            pipeline_name,
            context,
            self._gangs,
        )
        self.hf_workers.append(hf_worker)

        log.info(f"Replica {replica_i} setup completed")

    def is_model_ready(self):
        ready_refs = [llm.is_ready.remote() for llm in self.hf_workers]
        return ready_refs

    def setup_hf_worker(
        self,
        ray_actor_name,
        tensor_parallel_size,
        pipeline_name: str,
        context: RuntimeContext,
        gangs: Gangs,
    ):

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

        unit_handlers = context.get_registry(RemoteModelHandler)
        pipeline = unit_handlers.get(pipeline_name)
        llm = pipeline.options(
            name=ray_actor_name,
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=scheduling_inference,
            get_if_exists=True,
        ).remote()

        if self.blocking_initialization:
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


class RemoteModelHandler(ABC):
    @abstractmethod
    def create(
        self, gangs: Gangs, unit_config: object
    ) -> Union[RemoteVllmModel, RemoteHFModel]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class RemoteRayModelHandler(RemoteModelHandler):
    @override
    def create(
        self, gangs: Gangs, actor_config: RayActorConfig, context: RuntimeContext
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
                    actor_config.blocking_initialization,
                    context,
                    gangs,
                )
            else:
                actor_config = structure(actor_config, HFRayActorConfig)
                remote_vllm_model = RemoteHFModel(
                    actor_config.ray_actor_name,
                    actor_config.num_replicas,
                    actor_config.tensor_parallel_size,
                    actor_config.pipeline_name,
                    context,
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
