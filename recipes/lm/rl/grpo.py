# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, final
import ray

from torch import Tensor
import torch
from typing_extensions import override

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import SequenceBatch
from fairseq2.metrics import MetricBag
from fairseq2.metrics.common import (
    add_nll_loss_metric,
    add_seq_batch_metrics,
    update_nll_loss_metric,
    update_seq_batch_metrics,
)
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.trainer import Trainer, TrainUnit
from fairseq2.runtime.dependency import DependencyContainer
from fairseq2.gang import Gangs, Gang

from .utils import StatefulRolloutBag, generate_rollouts, log_rollouts

from ..common import check_vocab_info
from .dataset import (
    PromptBatch,
    LM_RL_DATASET,
    collate_with_target_mask
)
from .remote_model import RemoteVllmModel, maybe_sync_model

from .rewards.utils import Reward
from .rewards.math_verify_reward import MathVerifyVerifier
from .config import GrpoUnitConfig

GRPO_FINETUNE_UNIT = "grpo"

@dataclass
class GRPOBatch:
    """Represents a preference optimization dataset batch."""

    prompt_rollouts: SequenceBatch
    prompt_lengths: list[int]
    rewards: torch.Tensor

@final
class GRPOTrainUnit(TrainUnit[PromptBatch]):
    def __init__(self, model, remote_policy_model: RemoteVllmModel, remote_reference_model: RemoteVllmModel, reward: Reward, config: GrpoUnitConfig, gangs: Gangs) -> None:
        self._model = model
        self._remote_reference_model = remote_reference_model
        self._remote_policy_model = remote_policy_model
        self._config = config
        
        self._gangs = gangs
        self._reward = reward
        self._rollout_bag = StatefulRolloutBag(
            max_bag_steps=int(
                config.loss_config.group_size / config.loss_config.forward_group_size
            )
        )

        self._display_name = "GRPO"

    @override
    def set_step_nr(self, step_nr: int) -> None:
        self._step_nr = step_nr


    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_nll_loss_metric(metric_bag)
        add_seq_batch_metrics(metric_bag)

    def maybe_sync_models(self):
        for remote_model in [
            self._remote_policy_model,
            self._remote_reference_model
        ]:
            maybe_sync_model(
                self._gangs,
                self._model,
                remote_model,
                self._step_nr,
                remote_model.sync_every_n_steps,
            )

    @override
    def process_batch(
        self, prompt_batch: PromptBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, None]:
        
        self._rollout_bag.maybe_reset_bag(self._step_nr)

        if len(self._rollout_bag) == 0:
            
            self.maybe_sync_models()

            rollouts = generate_rollouts(
                prompt_batch.prompts,
                dp_gang=self._gangs.dp,
                remote_model=self._remote_policy_model,
            )
            if self._config.loss_config.log_rollouts:
                log_rollouts(prompt_batch, rollouts, "Train")

            reward_output = self._reward.process_rollouts(prompt_batch, rollouts)
            self._rollout_bag.save(rollouts, reward_output)

        else:
            rollouts, reward_output = self._rollout_bag.load()

        import ipdb; ipdb.set_trace()

        nll_loss = self._model.module(
            seqs, seqs_layout, targets=target_batch.seqs, reduction="mean"
        )

        update_nll_loss_metric(metric_bag, nll_loss)
        update_seq_batch_metrics(metric_bag, prompt_batch)

        return nll_loss, None

    def prepare_grpo_batch(
        self,
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
            prompt_rollouts=prompt_rollout_batch,
            rewards=rewards_normalized,
            prompt_lengths=prompt_lens,
        )

        return grpo_batch

    @property
    @override
    def model(self) -> RecipeModel:
        return self._model


def create_grpo_unit(
        model,
        gangs,
        config: GrpoUnitConfig,
        ray_actors: dict,
) -> GRPOTrainUnit:
    
    remote_policy_model = ray_actors[config.remote_policy_model_name]
    remote_reference_model = ray_actors[config.remote_reference_model_name]

    reward_config = config.reward.config
    reward = MathVerifyVerifier(
        answer_key=reward_config.answer_key,
        prompt_key=reward_config.prompt_key,
        reward_name="math_verify",
        gangs=gangs
    )

    return GRPOTrainUnit(model, remote_policy_model=remote_policy_model, remote_reference_model=remote_reference_model, reward=reward, config=config, gangs=gangs)