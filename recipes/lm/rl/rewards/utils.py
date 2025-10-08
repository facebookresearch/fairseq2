from dataclasses import dataclass, field
from typing import List, Protocol, cast

from ..dataset import PromptBatch, collate_with_target_mask
from fairseq2.datasets import SequenceBatch
from fairseq2.data.data_pipeline import SequenceData, Collater, CollateOptionsOverride
import torch

from vllm import RequestOutput

# from ..config import RegimeSection, RewardModelConfig

class Reward(Protocol):
    def process_rollouts(self, prompt_batch: PromptBatch, vllm_outputs: list[RequestOutput]): ...

def find_first_value(lst, value):
    return next((i for i, x in enumerate(lst) if x == value), None)

def prepare_preference_batch_random_pair(
    prompt_batch: PromptBatch, reward_output: dict, gangs
) -> dict[str, SequenceBatch | bool | list[int]]:
    """
    Single & random preference pair from rollouts and rewards
    """

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

    to_return = {
        "chosen_batch": chosen_batch,
        "rejected_batch": rejected_batch,
        "is_bad_batch": is_bad_batch,
        "dummy_batch_ids": dummy_batch_ids,
        "prompt_lengths": prompt_lens
    }

    return to_return


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