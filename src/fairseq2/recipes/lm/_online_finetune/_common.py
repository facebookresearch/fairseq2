# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import List, cast

import ray
import torch
import torch.nn as nn
from torch import Tensor
from vllm import RequestOutput

from fairseq2.data import CollateOptionsOverride, Collater, SequenceData
from fairseq2.datasets import SequenceBatch
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.metrics import Mean, MetricBag, Sum
from fairseq2.nn._batch_layout import BatchLayout
from fairseq2.nn.utils.padding import pad_seqs
from fairseq2.recipes.lm._online_finetune._remote_model import RemoteVllmModel


@dataclass(kw_only=True)
class OnlineCriterionSection:
    name: str
    config: object


@dataclass(kw_only=True)
class VllmSyncSection:
    sync_model_every_n_steps: int = 1
    """How often to sync the vLLM model with the policy that is trained. -1 disables syncing."""

    sync_ref_model_every_n_steps: int = -1
    """How often to sync the reference model with the policy. -1 disables syncing."""


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


def get_ray_actor(gangs: Gang, actor_name):
    # only retrieve vllm actors on main rank process as a safety measure to avoid blocking
    if gangs.dp.rank == 0 and gangs.tp.rank == 0:
        actor = ray.get_actor(actor_name)
    else:
        actor = None

    return actor


def collate_with_target_mask(
    list_of_tensors, prompt_lengths, pad_value=0, device="cpu"
):
    # list_of_tensors contain prompt+rollout tokens, we use prompt_len to define the target loss mask here
    to_collate = []
    for seq, prompt_len in zip(list_of_tensors, prompt_lengths):
        target_loss_mask = torch.arange(len(seq)) >= prompt_len
        to_collate.append({"seqs": seq, "target_loss_mask": target_loss_mask})

    target_mask_collate_opts = [
        CollateOptionsOverride("target_loss_mask", pad_value=False),
    ]
    collater = Collater(
        pad_value=pad_value, pad_to_multiple=1, overrides=target_mask_collate_opts
    )
    # from fairseq2.utils.env import get_rank
    # from os import environ
    # if get_rank(environ) == 0:
    #     import ipdb; ipdb.set_trace()
    # torch.distributed.barrier()

    seq_data = cast(SequenceData, collater(to_collate))

    batch = SequenceBatch(
        seq_data["seqs"]["seqs"],
        seq_data["seqs"]["seq_lens"],
        target_mask=seq_data["target_loss_mask"]["seqs"],
    )
    batch.to(device)

    return batch


def copy_state(src_module: nn.Module, tgt_module: nn.Module):
    tgt_state = tgt_module.state_dict()  # assumed tgt is not sharded
    for name, src_param in src_module.named_parameters():
        name_edited = name.replace("_checkpoint_wrapped_module.", "")
        if name_edited not in tgt_state.keys():
            raise NameError(f"{name_edited} doesnt exist in tgt_module")
        tgt_param = tgt_state[name_edited]
        tgt_param.data.copy_(src_param.data.to(tgt_param.device))


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
            prompt_list=flat_request_list, sampling_params=sampling_params
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


def generate_rewards_generative(
    prompts: List[List[int]],
    dp_gang,
    vllm_model,
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

        rewards = vllm_model.rollout_from_model(flat_request_list, string_input=True)

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

    return batch, is_bad_batch, dummy_batch_ids


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


def get_vllm_logprobs(
    vllm_outputs: List[RequestOutput],
    model_logps: Tensor,
    gangs,
    rollout_start_end: tuple[int, int] | None = None,
):
    """Compute per-token logprobs for selected continuations across a list of requests.

    For each RequestOutput (one prompt) and each of its sampled continuations we
    concatenate the prompt logprobs (skipping the first entry) with the generation
    logprobs. All resulting sequences are then right-padded with 0.0 to the global
    maximum length and stacked into a single tensor.

    Parameters
    ----------
    vllm_outputs:
        List of vLLM RequestOutput objects (one per prompt).
    gangs:
        Fairseq2 gangs object (unused, kept for parity/extensibility).
    rollout_start_end:
        Optional (start, end) slice specifying which continuation indices to include
        per prompt (used for micro-batching when forward_group_size < group_size).

    Returns
    -------
    Tensor
        Shape ``(num_selected_continuations, max_seq_len)`` with 0.0 padding.
    """
    sequences: List[Tensor] = []
    for request in vllm_outputs:
        prompt_logprobs = [
            list(d.values())[0].logprob for d in request.prompt_logprobs[1:]
        ]
        outputs = request.outputs
        if rollout_start_end is not None:  # micro-batching
            s, e = rollout_start_end
            outputs = outputs[s:e]
        for output in outputs:
            gen_logprobs = [list(d.values())[0].logprob for d in output.logprobs]
            seq = torch.tensor(prompt_logprobs + gen_logprobs)
            sequences.append(seq)

    max_len = max(t.size(0) for t in sequences)
    padded = torch.zeros(len(sequences), max_len)
    for i, t in enumerate(sequences):
        padded[i, : t.size(0)] = t

    # clip outputs to be same size as model_logps
    if padded.size() != model_logps.size():
        padded = padded[:, : model_logps.size(1)]
    return padded


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
    """Calculate entropy from logits. Returns sum of entropies averages for each sequence.

    Uses numerically stable entropy calculation: H = log(Z) - E[logits]
    where Z = sum(exp(logits)) and E is the expectation over softmax distribution.

    Note: Entropy is computed without gradients to avoid numerical instability.
    Memory-efficient implementation processes sequences in chunks.
    """
    # Compute entropy without gradients to avoid backprop instability
    with torch.no_grad():
        logits_clamped = torch.clamp(logits, min=-100, max=100)

        # Compute logsumexp for normalization constant
        log_z = torch.logsumexp(logits_clamped, dim=-1)  # [batch, seq_len]

        # Compute expected logits: E[logits] = sum(softmax(logits) * logits)
        # Process in smaller chunks to avoid OOM
        batch_size, seq_len, vocab_size = logits_clamped.shape
        chunk_size = 64  # Process 64 positions at a time

        expected_logits = torch.zeros(batch_size, seq_len, device=logits.device)
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            chunk_logits = logits_clamped[:, i:end_i, :]
            chunk_probs = torch.softmax(chunk_logits, dim=-1)
            expected_logits[:, i:end_i] = torch.sum(chunk_probs * chunk_logits, dim=-1)
            del chunk_probs, chunk_logits  # Free memory immediately

        # Entropy = log(Z) - E[logits]
        entropy = log_z - expected_logits

        # Clamp entropy to prevent NaN values
        entropy = torch.clamp(entropy, min=0.0, max=100.0)

    entropy_target_only = entropy * target_mask

    # Compute target token counts and avoid division by zero
    target_counts = target_mask.sum(dim=-1)
    # Add small epsilon to avoid division by zero
    target_counts = torch.clamp(target_counts, min=1e-8)
    entropy_per_seq = entropy_target_only.sum(dim=-1) / target_counts

    return entropy_per_seq


def log_rollouts(prompt_batch: PromptBatch, rollouts, split_name, num_rollouts=1):
    """
    log the first num_rollouts rollouts for first prompt in the batch
    """
    if "prompt_raw" in prompt_batch.meta_info:
        prompt = prompt_batch.meta_info.get("prompt_raw")[0]
    elif "raw_prompt" in prompt_batch.meta_info:
        prompt = prompt_batch.meta_info.get("raw_prompt")[0]
    elif "raw_prompt_text" in prompt_batch.meta_info:
        prompt = prompt_batch.meta_info.get("raw_prompt_text")[0]
    elif "prefix_text" in prompt_batch.meta_info:
        prompt = prompt_batch.meta_info.get("prefix_text")[0]
    else:
        # raw text prompt doesn't exist for this dataset
        prompt = "DUMMY PROMPT"

    log.info(f"{split_name} Prompt: {prompt}")
    for rollout in rollouts[0].outputs[:num_rollouts]:
        rollout_text = rollout.text
        log.info(f"{split_name} Rollout: {rollout_text}")


def get_rollout_lengths(rollouts: List[SequenceData]):
    """Get the lengths of the rollouts."""
    rollout_lengths = []
    for rollout in rollouts:
        for sample in rollout.outputs:
            token_ids = sample.token_ids
            token_ids_len = len(token_ids)
            rollout_lengths.append(token_ids_len)
    return rollout_lengths


def get_think_rollout_lengths(rollouts: List[SequenceData]):
    """Get the lengths of tokens before the </think> tag in rollouts.

    This function calculates the approximate number of tokens generated before
    the </think> closing tag in each rollout. It uses a proportional approximation
    based on character positions to estimate token counts.

    Args:
        rollouts: List of SequenceData containing rollout outputs

    Returns:
        List of token lengths before </think> tag for rollouts that contain the tag
    """
    think_rollout_lengths = []
    think_tag = "</think>"

    for rollout in rollouts:
        for sample in rollout.outputs:
            rollout_text = sample.text
            if think_tag in rollout_text:
                # Find the position of </think> in the text
                think_end_pos = rollout_text.find(think_tag) + len(think_tag)
                # Count tokens up to and including </think>
                # We need to find how many tokens correspond to the text before </think>
                # Since we have token_ids, we'll approximate by finding the proportion
                text_before_think = rollout_text[:think_end_pos]
                total_text = rollout_text
                total_tokens = len(sample.token_ids)
                # Approximate token count proportionally (rough estimate)
                # A better approach would be to tokenize text_before_think, but we use approximation
                think_token_length = (
                    int((len(text_before_think) / len(total_text)) * total_tokens)
                    if len(total_text) > 0
                    else 0
                )
                think_rollout_lengths.append(think_token_length)

    return think_rollout_lengths


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

    def __init__(self, max_bag_steps):
        self.rollouts: List = []
        self.reward_outputs: List = []
        self.max_bag_steps = max_bag_steps

    def maybe_reset_bag(self, trainer_step):
        # this is called every train step to see if we need to reset the bag
        if self.bag_step == self.max_bag_steps:
            # new trainer step, reset bag and counters
            self.rollouts = []
            self.reward_outputs = []
            self.bag_step = 0
            self._trainer_step = trainer_step

    def __len__(self):
        return len(self.rollouts)

    def save(self, rollouts, reward_outputs):
        self.rollouts = rollouts
        self.reward_outputs = reward_outputs
        self.bag_step += 1

    def load(self):
        self.bag_step += 1
        return self.rollouts, self.reward_outputs

    def get_rollout_start_end(self, num_rollout_per_forward: int):
        start_i = (self.bag_step - 1) * num_rollout_per_forward
        end_i = start_i + num_rollout_per_forward
        return start_i, end_i


@torch.inference_mode()
def update_logit_entropy(metric_bag: MetricBag, logit_entropy: Tensor):
    # logit_entropy is expected to contain token-level entropy for every sequence in the current batch
    batch_size = logit_entropy.size(0)
    metric_bag.get(Mean, "logit_entropy").update(
        logit_entropy.sum() / batch_size, weight=batch_size
    )


@torch.inference_mode()
def update_dpo_loss(metric_bag: MetricBag, loss: Tensor, batch_size: int) -> None:
    metric_bag.get(Mean, "dpo_loss").update(loss / batch_size, weight=batch_size)


@torch.inference_mode()
def update_num_dummy_batches(
    metric_bag: MetricBag, batch: PreferenceBatch, num_dummy_batches: int
):
    metric_bag.get(Mean, "num_dummy_batches").update(
        num_dummy_batches / batch.chosen.batch_size, weight=batch.chosen.batch_size
    )


@torch.inference_mode()
def update_avg_reward(metric_bag: MetricBag, avg_reward):
    metric_bag.get(Mean, "avg_reward").update(avg_reward, weight=1)


@torch.inference_mode()
def update_std_reward(metric_bag: MetricBag, std_reward):
    metric_bag.get(Mean, "std_reward").update(std_reward, weight=1)


@torch.inference_mode()
def update_avg_rollout_length(metric_bag: MetricBag, avg_rollout_length):
    metric_bag.get(Mean, "avg_rollout_length").update(avg_rollout_length, weight=1)


@torch.inference_mode()
def update_avg_think_rollout_length(metric_bag: MetricBag, avg_think_rollout_length):
    metric_bag.get(Mean, "avg_think_rollout_length").update(
        avg_think_rollout_length, weight=1
    )


@torch.inference_mode()
def update_mean_tok_cov(metric_bag: MetricBag, mean_tok_cov):
    metric_bag.get(Mean, "mean_tok_cov").update(mean_tok_cov, weight=1)


@torch.inference_mode()
def update_cov_clip_ratio(metric_bag: MetricBag, cov_clip_ratio):
    metric_bag.get(Mean, "cov_clip_ratio").update(cov_clip_ratio, weight=1)


@torch.inference_mode()
def update_avg_reward_len_norm(metric_bag: MetricBag, avg_reward_len_norm):
    metric_bag.get(Mean, "avg_reward_len_norm").update(avg_reward_len_norm, weight=1)


@torch.inference_mode()
def update_avg_loss_zeroer(metric_bag: MetricBag, avg_loss_zeroer):
    metric_bag.get(Mean, "avg_loss_zeroer").update(avg_loss_zeroer, weight=1)


@torch.inference_mode()
def update_batch_metrics(metric_bag: MetricBag, batch: PreferenceBatch, train: bool):
    num_examples = batch.batch_size
    metric_bag.get(Sum, "num_examples").update(num_examples)
    if train:
        metric_bag.get(Sum, "total_num_examples").update(num_examples)


def update_grpo_batch_metrics(
    metric_bag: MetricBag, batch: SequenceBatch, train=True
) -> None:
    metric_bag.get(Sum, "num_examples").update(batch.num_examples)

    metric_bag.get(Sum, "num_elements").update(batch.num_elements)

    metric_bag.get(Sum, "num_target_elements").update(batch.num_target_elements)

    metric_bag.get(Sum, "padding").update(batch.padding)

    if train:
        metric_bag.get(Sum, "total_num_examples").update(batch.num_examples)

        metric_bag.get(Sum, "total_num_elements").update(batch.num_elements)

        metric_bag.get(Sum, "total_num_target_elements").update(
            batch.num_target_elements
        )


@torch.inference_mode()
def update_grpo_loss(
    metric_bag: MetricBag, batch: PromptBatch, loss: Tensor, tis_imp_ratio: Tensor
) -> None:
    """Update the GRPO loss metric.

    :param batch:
        The batch processed by the model.
    :param loss:
        The GRPO loss of ``batch``.
    """
    metric_bag.get(Mean, "grpo_loss").update(
        loss / batch.batch_size, weight=batch.batch_size
    )

    metric_bag.get(Mean, "tis_imp_ratio").update(tis_imp_ratio)


def compute_reference_logps(
    gangs: Gangs,
    reference_model: RemoteVllmModel,
    seqs: torch.Tensor,
    layout: BatchLayout,
    prompt_lengths: list[int],
):

    seqs_to_score = seqs.tolist()
    if layout.padded:
        padding_mask = layout.position_indices >= 0  # True when non-pad
        seqs_to_score = [
            seq[:l] for seq, l in zip(seqs_to_score, padding_mask.sum(-1).tolist())
        ]

    scored_responses = generate_rollouts(
        seqs_to_score, dp_gang=gangs.dp, vllm_model=reference_model
    )
    ref_logps = convert_vllm_output_to_ref_score(scored_responses, gangs)
    ref_logps = collate_with_target_mask(
        ref_logps, prompt_lengths, device=gangs.dp.device
    ).seqs

    return ref_logps


def get_parameter_converter(model_config):

    from fairseq2.models.llama import LLaMAConfig
    from fairseq2.models.qwen import QwenConfig

    if isinstance(model_config, QwenConfig):
        from fairseq2.models.qwen._hg import _convert_parameter
    elif isinstance(model_config, LLaMAConfig):
        from fairseq2.models.llama._hg import _convert_parameter
    else:
        raise RuntimeError(f"{model_config} not supported in online recipe")

    return _convert_parameter
