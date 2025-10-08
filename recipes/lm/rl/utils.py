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

from fairseq2.runtime.dependency import DependencyContainer

from fairseq2.recipe.trainer import TrainUnit

from fairseq2.runtime.dependency import DependencyResolver, wire_object

from .dataset import PromptBatch

from fairseq2.logging import log

def log_rollouts(prompt_batch: PromptBatch, rollouts, split_name, num_rollouts=1):
    """
    log the first num_rollouts rollouts for first prompt in the batch
    """
    if "prompt_raw" in prompt_batch.meta_info:
        prompt = prompt_batch.meta_info.get("prompt_raw")[0]
    elif "raw_prompt" in prompt_batch.meta_info:
        prompt = prompt_batch.meta_info.get("raw_prompt")[0]
    else:
        # raw text prompt doesn't exist for this dataset
        prompt = "DUMMY PROMPT"

    log.info(f"{split_name} Prompt: {prompt}")
    for rollout in rollouts[0].outputs[:num_rollouts]:
        rollout_text = rollout.text
        log.info(f"{split_name} Rollout: {rollout_text}")

def generate_rollouts(
    prompts: List[List[int]],
    dp_gang,
    remote_model,
    sampling_params=None,
) -> list[RequestOutput]:
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

        rollouts = remote_model.rollout_from_model(
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
    _trainer_step: int | None = None

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


def register_rl_train_unit(
    container: DependencyContainer,
    name: str,
    kls: type[TrainUnit],
    config_kls: object,
    factory: object,
) -> None:
    if factory is None:
        raise ValueError("`factory` must be specified.")

    def create_unit(resolver: DependencyResolver) -> TrainUnit:
        nonlocal factory

        return wire_object(
            resolver,
            TrainUnit,
            name=name,
            kls=kls,
            config_kls=config_kls,
            loader=factory,
        )

    container.register(TrainUnit, create_unit, key=name)

def get_parameter_converter(model_config):

    from fairseq2.models.llama import LLaMAConfig
    from fairseq2.models.qwen import QwenConfig

    if isinstance(model_config, QwenConfig):
        from fairseq2.models.qwen.interop import convert_parameter
    elif isinstance(model_config, LLaMAConfig):
        from fairseq2.models.llama.interop import convert_parameter
    else:
        raise RuntimeError(f"{model_config} not supported in RL recipe")

    return convert_parameter
