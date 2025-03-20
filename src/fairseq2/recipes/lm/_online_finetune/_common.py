# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, List

import torch.nn as nn

import torch
from torch import Tensor
from torcheval.metrics import Mean

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch, SequenceModelOutput
from fairseq2.recipes.metrics import SequenceMetricBag

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput, TokensPrompt
import re
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker
import os

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.nn.padding import get_seqs_and_padding_mask


@dataclass
class GRPOBatch:
    """Represents a preference optimization dataset batch."""

    prompt_rollouts: SequenceBatch
    rewards: torch.Tensor


@dataclass(kw_only=True)
class OnlineCriterionSection:
    name: str

    config: object


class OnlineFinetuneMetricBag(SequenceMetricBag):

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)


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
        super().__init__(*args, **kwargs)

        self.ready = True  # Set a flag or return a signal

    def is_ready(self):
        return self.ready


class MyWorker(Worker):
    """
    The `MyWorker` class inherits from `Worker` to provide custom functions.
    For simplicity, we define the `MyWorker` class in this self-contained
    script. Normally, we should define the `MyWorker` class in a separate
    file and pass the qualified name of the class to the `worker_cls`
    parameter.
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

        # wrap in fs2 style dict
        weights = {"model_key": "model", "model": {name: weight}}.items()
        # self.model_runner.model.load_weights(weights=[(name, weight)])
        self.model_runner.model.load_weights(weights=weights)

        del weight


def setup_vllm(
    actor_name,
    vllm_init_checkpoint_dir,
    vllm_init_tokenizer,
    tensor_parallel_size,
    dp_device,
):

    pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * tensor_parallel_size)

    ray.get(pg_inference.ready())

    scheduling_inference = PlacementGroupSchedulingStrategy(
        placement_group=pg_inference,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )

    """
    launch the vLLM inference engine.
    here we use `enforce_eager` to reduce the start time.
    """
    llm = NoEnvLLM.options(
        name=actor_name,
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,
        get_if_exists=True,
    ).remote(
        model=vllm_init_checkpoint_dir,
        tokenizer=vllm_init_tokenizer,
        enforce_eager=True,
        worker_cls=MyWorker,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="ray",
    )

    # we block here until the engine is initialized
    ray.get(llm.is_ready.remote())

    # setting up process groups

    master_port = get_open_port()
    master_address = get_ip()

    print(f"{master_port} {master_address}")

    print("init pg on vllm host")
    handle = llm.collective_rpc.remote(
        "init_weight_update_group",
        args=(master_address, master_port, 1, tensor_parallel_size + 1),
    )

    print("init pg on train host")
    model_update_group = stateless_init_process_group(
        master_address, master_port, 0, tensor_parallel_size + 1, dp_device
    )
    ray.get(handle)

    return llm, model_update_group


def gsm8k_correctness_verifier(vllm_output: RequestOutput, reference_answer: List[str]):
    # verifier to match predicted answer with gsm8k format with the reference

    # utils from gsm8k paper to extract a correct answer and match it to the prompt
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"

    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS

    batch_text = []
    batch_tokens = []
    batch_rewards = []

    for i, i_batch_request_output in enumerate(vllm_output):
        rollouts_text = []
        rollouts_tokens = []
        i_reference_answer = reference_answer[i]
        rollouts_rewards = []
        for rollout_output in i_batch_request_output.outputs:
            rollouts_text.append(rollout_output.text)
            rollouts_tokens.append(rollout_output.token_ids)
            predicted_answer = extract_answer(rollout_output.text)
            predicted_reward = 1 if predicted_answer == i_reference_answer else 0
            rollouts_rewards.append(predicted_reward)
        batch_text.append(rollouts_text)
        batch_tokens.append(rollouts_tokens)
        batch_rewards.append(rollouts_rewards)

    return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}


def collate_with_target_mask(list_of_tensors, prompt_lens, pad_value=0, device="cpu"):
    # list_of_tensors contain prompt+rollout tokens, we use prompt_len to define the target loss mask here
    to_collate = []
    for seq, prompt_len in zip(list_of_tensors, prompt_lens):
        target_loss_mask = torch.arange(len(seq)) >= prompt_len
        to_collate.append({"seqs": seq, "target_loss_mask": target_loss_mask})

    target_mask_collate_opts = [
        CollateOptionsOverride("target_loss_mask", pad_value=False),
    ]
    collater = Collater(
        pad_value=pad_value, pad_to_multiple=1, overrides=target_mask_collate_opts
    )

    seq_data = cast(SequenceData, collater(to_collate))

    seqs, padding_mask = get_seqs_and_padding_mask(seq_data["seqs"], device)

    batch = SequenceBatch(
        seqs=seqs,
        padding_mask=padding_mask,
        target_mask=seq_data["target_loss_mask"]["seqs"].to(device),
    )

    return batch


def copy_state(src_module: nn.Module, tgt_module: nn.Module):
    tgt_state = tgt_module.state_dict()  # assumed tgt is not sharded
    for name, src_param in src_module.named_parameters():
        name_edited = name.replace("_checkpoint_wrapped_module.", "")
        if name_edited not in tgt_state.keys():
            raise NameError(f"{name_edited} doesnt exist in tgt_module")
        tgt_param = tgt_state[name_edited]
        tgt_param.data.copy_(src_param.data.to(tgt_param.device))


def sync_weights_with_vllm(train_model, vllm_model, trainer_process_group):
    """
    trainer_process_group must connect training process with vllm_model processes
    """
    for name, p in train_model.module.named_parameters():
        name = name.replace("._checkpoint_wrapped_module", "")
        # print(f'sync call {name}')
        handle = vllm_model.collective_rpc.remote(
            "update_weight", args=(name, p.dtype, p.shape)
        )
        trainer_process_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
        ray.get(handle)


def find_first_value(lst, value):
    return next((i for i, x in enumerate(lst) if x == value), None)


def generate_rollouts(
    prompts: List[str], dp_gang: Gang, vllm_model, sampling_params=None
):
    prompts_to_generate = [None] * dp_gang.size
    if dp_gang.rank == 0:
        dp_gang.gather_object(prompts, prompts_to_generate, 0)
    else:
        dp_gang.gather_object(prompts, None, 0)

    if dp_gang.rank == 0:
        rank_batch_sizes = [len(l) for l in prompts_to_generate]
        flat_request_list = []
        for rank_prompts in prompts_to_generate:
            flat_request_list.extend(rank_prompts)

        rollouts = vllm_model.rollout_from_model(
            flat_request_list, sampling_params=sampling_params
        )

        rollouts_to_scatter = []
        rollouts_per_rank = [None]
        for dp_rank, rank_batch_size in zip(range(dp_gang.size), rank_batch_sizes):
            rank_start = sum(rank_batch_sizes[:dp_rank])
            rank_end = rank_start + rank_batch_size
            rollouts_to_scatter.append(rollouts[rank_start:rank_end])
        dp_gang.scatter_object_list(
            rollouts_per_rank, rollouts_to_scatter, source_rank=0
        )
    else:
        rollouts_per_rank = [None]
        dp_gang.scatter_object_list(rollouts_per_rank, None, source_rank=0)

    dp_gang.barrier()

    return rollouts_per_rank[0]


def generate_rewards(
    prompts: List[str], dp_gang: Gang, vllm_model, sampling_params=None
):
    # FIXME should be combined with generate_rollouts
    prompts_to_generate = [None] * dp_gang.size
    if dp_gang.rank == 0:
        dp_gang.gather_object(prompts, prompts_to_generate, 0)
    else:
        dp_gang.gather_object(prompts, None, 0)

    if dp_gang.rank == 0:

        rank_batch_sizes = [len(l) for l in prompts_to_generate]
        flat_request_list = []
        for rank_prompts in prompts_to_generate:
            flat_request_list.extend(rank_prompts)

        rollouts = vllm_model.get_reward_from_model(
            flat_request_list, sampling_params=sampling_params
        )

        rollouts_to_scatter = []
        rollouts_per_rank = [None]
        for dp_rank, rank_batch_size in zip(range(dp_gang.size), rank_batch_sizes):
            rank_start = sum(rank_batch_sizes[:dp_rank])
            rank_end = rank_start + rank_batch_size
            rollouts_to_scatter.append(rollouts[rank_start:rank_end])
        dp_gang.scatter_object_list(
            rollouts_per_rank, rollouts_to_scatter, source_rank=0
        )
    else:
        rollouts_per_rank = [None]
        dp_gang.scatter_object_list(rollouts_per_rank, None, source_rank=0)

    dp_gang.barrier()

    return rollouts_per_rank[0]
