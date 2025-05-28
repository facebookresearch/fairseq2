# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, List, cast

import ray
import torch
import torch.nn as nn
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import Tensor
from torcheval.metrics import Mean
from transformers import AutoModelForCausalLM
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch, SequenceModelOutput
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.recipes.metrics import SequenceMetricBag
from fairseq2.logging import log


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


def get_ray_actor(gangs: Gang, actor_name):
    # only retrieve vllm actors on main rank process as a safety measure to avoid blocking
    if gangs.dp.rank == 0 and gangs.tp.rank == 0:
        actor = ray.get_actor(actor_name)
    else:
        actor = None

    return actor


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
        os.environ["VLLM_USE_V1"] = "0"
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
    prompts: List[List[int]],
    dp_gang,
    vllm_model,
    sampling_params=None,
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
    prompts: List[List[int]],
    dp_gang,
    vllm_model,
    sampling_params=None,
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

        rewards = vllm_model.reward_from_model(flat_request_list)

        rewards_to_scatter = []
        rewards_per_rank = [None]
        for dp_rank, rank_batch_size in zip(range(dp_gang.size), rank_batch_sizes):
            rank_start = sum(rank_batch_sizes[:dp_rank])
            rank_end = rank_start + rank_batch_size
            rewards_to_scatter.append(rewards[rank_start:rank_end])
        dp_gang.scatter_object_list(rewards_per_rank, rewards_to_scatter, source_rank=0)
    else:
        rewards_per_rank = [None]
        dp_gang.scatter_object_list(rewards_per_rank, None, source_rank=0)
    dp_gang.barrier()

    return rewards_per_rank[0]


def generate_rewards_generative(prompts: List[List[int]], dp_gang, vllm_model):
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

        rewards = vllm_model.reward_from_generative_model(flat_request_list)

        rewards_to_scatter = []
        rewards_per_rank = [None]
        for dp_rank, rank_batch_size in zip(range(dp_gang.size), rank_batch_sizes):
            rank_start = sum(rank_batch_sizes[:dp_rank])
            rank_end = rank_start + rank_batch_size
            rewards_to_scatter.append(rewards[rank_start:rank_end])
        dp_gang.scatter_object_list(rewards_per_rank, rewards_to_scatter, source_rank=0)
    else:
        rewards_per_rank = [None]
        dp_gang.scatter_object_list(rewards_per_rank, None, source_rank=0)
    dp_gang.barrier()

    return rewards_per_rank[0]


def prepare_preference_batch_random_pair(
    prompt_batch: PromptBatch, reward_output: dict, gangs
) -> PreferenceBatch:
    """
    Single & random preference pair from rollouts and rewards
    """

    # reward_output = self.process_rollouts(rollouts, prompt_batch.meta_info[self.answer_key])

    chosen_batch = []
    rejected_batch = []
    prompt_lens = []
    dummy_batch_ids = []  # keep posiitons of dummy pairs here

    # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
    for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
        zip(reward_output["rewards"], reward_output["tokens"])
    ):
        chosen_rollout_position = find_first_value(i_batch_rewards, 1)
        rejected_rollout_position = find_first_value(i_batch_rewards, 0)
        if chosen_rollout_position is None or rejected_rollout_position is None:
            # cant form preference pair when we dont have such rollouts
            # this will be dummy batch and we zero out loss
            chosen_rollout_position = 0
            rejected_rollout_position = 1
            dummy_batch_ids.append(i_batch)
        chosen_rollout_tokens = list(i_batch_tokens[chosen_rollout_position])
        rejected_rollout_tokens = list(i_batch_tokens[rejected_rollout_position])
        prompt_tokens = prompt_batch.prompts[i_batch]

        chosen_tokens = prompt_tokens + chosen_rollout_tokens
        chosen_batch.append(chosen_tokens)

        rejected_tokens = prompt_tokens + rejected_rollout_tokens
        rejected_batch.append(rejected_tokens)

        prompt_lens.append(len(prompt_tokens))

    filter_batch = lambda batch: [
        item for index, item in enumerate(batch) if index not in dummy_batch_ids
    ]

    if len(dummy_batch_ids) == len(reward_output["tokens"]):
        # entire batch does not have a valid preference pair
        # we use it as dummy batch and zero the loss in the end
        is_bad_batch = True
    else:
        # removing dummy pairs from the batch
        chosen_batch = filter_batch(chosen_batch)
        rejected_batch = filter_batch(rejected_batch)
        prompt_lens = filter_batch(prompt_lens)
        is_bad_batch = False

    prompt_lens = torch.tensor(prompt_lens)

    chosen_batch = [
        torch.tensor(sequence, device=gangs.dp.device) for sequence in chosen_batch
    ]
    chosen_batch = collate_with_target_mask(
        chosen_batch, prompt_lens, device=gangs.dp.device
    )

    rejected_batch = [
        torch.tensor(sequence, device=gangs.dp.device) for sequence in rejected_batch
    ]
    rejected_batch = collate_with_target_mask(
        rejected_batch, prompt_lens, device=gangs.dp.device
    )

    batch = PreferenceBatch(
        chosen=chosen_batch,
        rejected=rejected_batch,
        reference_score_chosen=None,
        reference_score_rejected=None,
    )

    return batch, is_bad_batch


def prepare_group_dpo_batch(
    prompt_batch: PromptBatch, reward_output: dict, gangs
) -> PreferenceBatch:
    """
    In group DPO we want to forward all rollouts, and then match all correct vs incorrect options in the loss
    """

    batch = []
    prompt_lens = []
    rewards = []
    dummy_batch_ids = []  # keep posiitons of dummy pairs here

    batch_size = prompt_batch.batch_size

    # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
    for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
        zip(reward_output["rewards"], reward_output["tokens"])
    ):
        if len(set(i_batch_rewards)) == 1:
            # same reward for all rollouts, we wont be able to use it for pairs
            dummy_batch_ids.append(i_batch)

        for rollout_tokens in i_batch_tokens:
            prompt_rollout_tokens = prompt_batch.prompts[i_batch] + list(rollout_tokens)
            batch.append(prompt_rollout_tokens)
            prompt_lens.append(len(prompt_batch.prompts[i_batch]))
        rewards.extend(i_batch_rewards)

    prompt_lens = torch.tensor(prompt_lens)

    batch = [torch.tensor(sequence, device=gangs.dp.device) for sequence in batch]
    batch = collate_with_target_mask(
        batch, prompt_lens, device=gangs.dp.device
    )  # [batch_size * n_rollout]
    rewards = torch.tensor(rewards, device=gangs.dp.device).view(
        batch_size, -1
    )  # [batch_size * n_rollout]

    return batch, rewards, dummy_batch_ids


def prepare_grpo_batch(
    prompt_batch: PromptBatch,
    reward_output: dict,
    gangs: Gang,
    rollout_start_end: tuple[int],
):

    prompt_rollouts = []
    prompt_lens = []
    rewards = []

    for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
        zip(reward_output["rewards"], reward_output["tokens"])
    ):
        prompt = prompt_batch.prompts[i_batch]
        rollout_tokens = [
            torch.tensor(prompt + list(c), device=gangs.dp.device)
            for c in i_batch_tokens[rollout_start_end[0] : rollout_start_end[1]]
        ]

        prompt_rollouts.extend(rollout_tokens)

        prompt_lens.extend([len(prompt)] * len(rollout_tokens))

        rewards.append(
            i_batch_rewards
        )  # we add all rewards here to correctly compute group statistic

    # if gangs.root.rank == 0:
    #     from pudb.remote import set_trace
    #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

    # gangs.root.barrier()

    rewards = torch.tensor(rewards, device=gangs.dp.device).float()  # [Batch, Rollouts]
    rewards_normalized = (rewards - rewards.mean(dim=1, keepdim=True)) / (
        rewards.std(dim=1, keepdim=True) + 1e-6
    )  # small epsilon to compensate 0 std

    rewards_normalized = rewards_normalized[
        :, rollout_start_end[0] : rollout_start_end[1]
    ]
    prompt_rollout_batch = collate_with_target_mask(
        prompt_rollouts, prompt_lens, device=gangs.dp.device
    )

    grpo_batch = GRPOBatch(
        prompt_rollouts=prompt_rollout_batch, rewards=rewards_normalized
    )

    return grpo_batch


def combine_prompts_responses_for_scoring(
    prompt_batch, rollouts: List[RequestOutput], gangs
):
    prompts: List[List[int]]
    prompts = prompt_batch.prompts

    responses = []
    for prompt, req_output in zip(prompts, rollouts):
        rollout_outputs = []
        for output in req_output.outputs:
            prompt_response_tokens = prompt + list(output.token_ids)
            rollout_outputs.append(prompt_response_tokens)
        responses.extend(rollout_outputs)

    # if gangs.root.rank == 0:
    #     from pudb.remote import set_trace
    #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

    # gangs.root.barrier()

    return responses


def convert_vllm_output_to_ref_score(vllm_outputs: List[RequestOutput], gangs):
    ref_scores = []
    for req_output in vllm_outputs:
        prompt_logprobs = req_output.prompt_logprobs[1:]
        logprobs = [list(d.values())[0].logprob for d in prompt_logprobs]
        # selecting only the response part that we scored
        logprobs = torch.tensor(logprobs)
        ref_scores.append(logprobs)

    return ref_scores


def compute_token_level_entropy(logits: torch.Tensor, target_mask: torch.Tensor):
    """Calculate entropy from logits. Returns sum of entropies averages for each sequence."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    entropy_target_only = entropy * target_mask
    entropy_per_seq = entropy_target_only.sum(dim=-1) / target_mask.sum(dim=-1)

    return entropy_per_seq


def log_rollouts(prompt_batch: PromptBatch, rollouts, split_name, num_rollouts=1):
    """
    log the first num_rollouts rollouts for first prompt in the batch
    """
    if "prompt_raw" in prompt_batch.meta_info:
        prompt = prompt_batch.meta_info.get("prompt_raw")[0]
    else:
        # raw text prompt doesn't exist for this dataset
        prompt = "DUMMY PROMPT"

    log.info(f"{split_name} Prompt: {prompt}")
    for rollout in rollouts[0].outputs[:num_rollouts]:
        rollout_text = rollout.text
        log.info(f"{split_name} Rollout: {rollout_text}")


class StatefulRolloutBag:
    """A stateful container for managing and reusing model rollouts across multiple micro-batches.

    This class enables efficient gradient accumulation in GRPO by:
    1. Generating rollouts once per training step
    2. Reusing these rollouts across multiple forward passes (micro-batches)
    3. Managing the windowing of rollouts for each micro-batch

    In GRPO training, generating rollouts is computationally expensive. When the group_size
    is large (many rollouts per prompt), processing all rollouts in a single forward pass
    may exceed memory limits. This class allows splitting the computation into smaller
    chunks by tracking which subset of rollouts should be used in each forward pass.

    Usage in GRPO:
    - At the beginning of each training step, call `maybe_reset_bag(step_nr)`
    - If bag is empty (first micro-batch of step), generate rollouts and save them
    - For subsequent micro-batches, reuse the same rollouts
    - Use `get_rollout_start_end()` to determine which slice of rollouts to process
      in the current micro-batch based on forward_group_size

    Attributes:
        bag_step: Current micro-batch step within the training step
        _trainer_step: Current training step
        rollouts: List of model rollouts generated for the current step
        reward_outputs: List of reward outputs for the rollouts
    """

    bag_step: int = 0
    _trainer_step: int = None

    def __init__(self):
        self.rollouts: List = []
        self.reward_outputs: List = []

    def maybe_reset_bag(self, trainer_step):
        # this is called every train step to see if we need to reset the bag
        if self._trainer_step != trainer_step:
            # new trainer step, reset bag and counters
            self.rollouts = []
            self.reward_outputs = []
            self.bag_step = 0
            self._trainer_step = trainer_step
        else:
            self.bag_step += 1

    def __len__(self):
        return len(self.rollouts)

    def save(self, rollouts, reward_outputs):
        self.rollouts = rollouts
        self.reward_outputs = reward_outputs

    def load(self):
        return self.rollouts, self.reward_outputs

    def get_rollout_start_end(self, num_rollout_per_forward: int):
        start_i = self.bag_step * num_rollout_per_forward
        end_i = start_i + num_rollout_per_forward
        return start_i, end_i
