# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Union, cast, final

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from fairseq2.context import RuntimeContext
from fairseq2.datasets import SequenceBatch
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.nn._batch_layout import BatchLayout
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.recipes import Model, TrainUnit
from fairseq2.recipes.lm._online_finetune._common import (
    StatefulRolloutBag,
    VllmSyncSection,
    collate_with_target_mask,
    compute_reference_logps,
    compute_token_level_entropy,
    generate_rollouts,
    get_rollout_lengths,
    get_vllm_logprobs,
    log_rollouts,
    update_avg_reward,
    update_avg_reward_len_norm,
    update_avg_rollout_length,
    update_batch_metrics,
    update_grpo_batch_metrics,
    update_grpo_loss,
    update_logit_entropy,
    update_std_reward,
)
from fairseq2.recipes.lm._online_finetune._handler import OnlineFinetuneUnitHandler
from fairseq2.recipes.lm._online_finetune._remote_model import (
    RemoteHFModel,
    RemoteVllmModel,
    maybe_sync_model,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    RewardSection,
    VLLMOutputReward,
    VLLMOutputRewardHandler,
)
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@dataclass
class GRPOBatch:
    """Represents a preference optimization dataset batch."""

    prompt_rollouts: SequenceBatch
    prompt_lengths: list[int]
    rewards: torch.Tensor


def prepare_grpo_batch(
    prompt_batch: PromptBatch,
    reward_output: dict,
    gangs: Gang,
    rollout_start_end: tuple[int],
    adv_std_normalization: bool,
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

    rewards_normalized = rewards - rewards.mean(dim=1, keepdim=True)
    if adv_std_normalization:  # normalize advantages with std
        rewards_normalized = rewards_normalized / (
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


@final
class GrpoFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _step_nr: int
    _valid_step_nr: int
    _reference_model: RemoteVllmModel
    _vllm_model: RemoteVllmModel
    _vllm_actors: Dict[str, Union[RemoteVllmModel, RemoteHFModel]]
    _config: GrpoFinetuneConfig
    _model_update_group: PyNcclCommunicator
    _reward: VLLMOutputReward
    _display_name: str
    _rollout_bag: StatefulRolloutBag

    def __init__(
        self,
        model: Module,
        reference_model: Module | RemoteVllmModel,
        vllm_model: RemoteVllmModel,
        vllm_actors: List[Union[RemoteVllmModel, RemoteHFModel]],
        reward,
        gangs: Gangs,
        config: GrpoFinetuneConfig,
    ) -> None:
        super().__init__()
        self._model = model
        self._reference_model = reference_model
        self._config = config
        self._vllm_actors = vllm_actors
        self._vllm_model = vllm_model
        self._gangs = gangs
        self._reward = reward
        self._rollout_bag = StatefulRolloutBag(
            max_bag_steps=int(
                config.loss_config.group_size / config.loss_config.forward_group_size
            )
        )

        self._display_name = "GRPO"

    @property
    @override
    def display_name(self) -> str | None:
        return self._display_name

    def set_train_step_nr(self, train_step_nr: int) -> None:
        self._valid_step_nr = train_step_nr

    def finalize(self, metric_bag: MetricBag) -> None:
        pass

    @property
    @override
    def name(self) -> str | None:
        return self._display_name

    def validate_reward(
        self, prompt_batch: PromptBatch, metric_bag
    ) -> tuple[Tensor, int]:
        if self._gangs.dp.rank == 0:
            policy_sampling_params = copy(self._vllm_model.sampling_params)
            # For a pairwise RM, need to sample at least two judgments
            policy_sampling_params.n = (
                2 if self._reward.reward_name == "generative_pairwise_verifier" else 1
            )
            for (
                k,
                v,
            ) in self._config.loss_config.validation_vllm_sampling_params.items():
                policy_sampling_params.__setattr__(k, v)
        else:
            policy_sampling_params = None
        rollouts = generate_rollouts(
            prompt_batch.prompts,
            dp_gang=self._gangs.dp,
            vllm_model=self._vllm_model,
            sampling_params=policy_sampling_params,
        )
        if self._config.loss_config.log_rollouts:
            log_rollouts(prompt_batch, rollouts, "Valid")
        reward_output = self._reward.process_rollouts(rollouts, prompt_batch)
        log.info(f"Rewards: {reward_output['rewards']}")
        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()
        std_reward = torch.tensor(reward_output["rewards"]).float().std()

        rollout_lengths = get_rollout_lengths(rollouts)
        avg_rollout_length = torch.tensor(rollout_lengths).float().mean()
        avg_reward_len_norm = avg_reward / avg_rollout_length

        update_avg_rollout_length(metric_bag, avg_rollout_length)
        update_avg_reward_len_norm(metric_bag, avg_reward_len_norm)

        update_avg_reward(metric_bag, avg_reward)
        update_batch_metrics(metric_bag, prompt_batch, train=False)
        update_std_reward(metric_bag, std_reward)
        # returning dummy loss since trainer expects it
        return torch.tensor(0.0, device=self._gangs.dp.device), prompt_batch.batch_size

    @override
    def __call__(
        self, prompt_batch: PromptBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:

        if not self.model.module.training:
            # we are in valid mode, only compute reward and return
            dummy_loss, batch_size = self.validate_reward(
                prompt_batch, metric_bag=metric_bag
            )
            return dummy_loss, batch_size

        self._rollout_bag.maybe_reset_bag(self._step_nr)

        if len(self._rollout_bag) == 0:

            maybe_sync_model(
                self._gangs,
                self._model,
                self._vllm_model,
                self._step_nr,
                self._config.vllm_sync.sync_model_every_n_steps,
            )
            maybe_sync_model(
                self._gangs,
                self._model,
                self._reference_model,
                self._step_nr,
                self._config.vllm_sync.sync_ref_model_every_n_steps,
            )

            rollouts = generate_rollouts(
                prompt_batch.prompts,
                dp_gang=self._gangs.dp,
                vllm_model=self._vllm_model,
            )
            if self._config.loss_config.log_rollouts:
                log_rollouts(prompt_batch, rollouts, "Train")

            reward_output = self._reward.process_rollouts(rollouts, prompt_batch)
            self._rollout_bag.save(rollouts, reward_output)

        else:
            rollouts, reward_output = self._rollout_bag.load()

        grpo_batch: GRPOBatch
        grpo_batch = prepare_grpo_batch(
            prompt_batch=prompt_batch,
            reward_output=reward_output,
            gangs=self._gangs,
            rollout_start_end=self._rollout_bag.get_rollout_start_end(
                self._config.loss_config.forward_group_size
            ),
            adv_std_normalization=self._config.loss_config.adv_std_normalization,
        )

        # grpo_batch, reward_output = self._reward.prepare_grpo_batch(prompt_batch, rollouts)  # loss_zeroer is used when entire batch has no valid prefrence pair

        (
            grpo_input_batch,
            grpo_target_batch,
        ) = grpo_batch.prompt_rollouts.as_auto_regressive()
        (
            grpo_input_batch_seqs,
            grpo_input_batch_seqs_layout,
        ) = grpo_input_batch.as_input()

        grpo_model_logits = self._model.module(
            grpo_input_batch_seqs, grpo_input_batch_seqs_layout
        )

        model_logps = self._gather_lprobs(grpo_model_logits, grpo_target_batch)
        vllm_logps = get_vllm_logprobs(rollouts, self._gangs).to(model_logps.device)

        tgt_logit_entropy = compute_token_level_entropy(
            grpo_model_logits, grpo_target_batch.target_mask
        )  # [Batch x Rollouts, 1]

        max_entropy_regularizer = (
            -tgt_logit_entropy.sum()
            * self._config.loss_config.entropy_regularizer_scale
        )
        update_logit_entropy(metric_bag, tgt_logit_entropy)

        (
            prompt_rollout_seqs,
            prompt_rollout_layout,
        ) = grpo_batch.prompt_rollouts.as_input()

        # if beta > 0, compute reference logprobs
        if self._config.loss_config.beta > 0:
            ref_logps = compute_reference_logps(
                self._gangs,
                self._reference_model,
                prompt_rollout_seqs,
                prompt_rollout_layout,
                grpo_batch.prompt_lengths,
            )
        else:
            ref_logps = None

        _grpo_objective, total_tokens = self._compute_grpo_objective(
            model_logps, vllm_logps, ref_logps, grpo_batch.rewards, grpo_target_batch
        )

        grpo_loss = -_grpo_objective + max_entropy_regularizer

        update_grpo_loss(metric_bag, prompt_batch, grpo_loss)

        rollouts_lengths = []
        for prompt_rollouts in reward_output["tokens"]:
            lengths = [len(r) for r in prompt_rollouts]  # lengths per the same prompt
            rollouts_lengths.append(lengths)
        rollouts_lengths = (
            torch.tensor(rollouts_lengths, device=self._gangs.dp.device)
            .float()
            .mean(dim=1)
        )  # [Batch]
        update_avg_rollout_length(metric_bag, rollouts_lengths)

        update_grpo_batch_metrics(
            metric_bag,
            grpo_batch.prompt_rollouts,
        )

        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()
        std_reward = torch.tensor(reward_output["rewards"]).float().std()

        update_std_reward(metric_bag, std_reward)
        update_avg_reward(metric_bag, avg_reward)

        loss = grpo_loss

        if self._config.loss_config.loss_token_mean:
            return loss, total_tokens
        else:
            return loss, prompt_batch.batch_size

    def _gather_lprobs(self, logits: Tensor, target: SequenceBatch) -> Tensor:
        assert target.target_mask is not None
        logprobs = torch.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(
            -1
        )  # [Batch, 1]

        return per_token_logps

    def _compute_grpo_objective(
        self,
        model_logps,
        vllm_logps,
        ref_logps,
        advantages: Tensor,  # outcome based only for now
        target_batch: SequenceBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:

        batch_size = advantages.size(0)
        num_rollouts = advantages.size(1)
        model_logps = model_logps.view(batch_size, num_rollouts, -1)
        vllm_logps = vllm_logps.view(batch_size, num_rollouts, -1)

        per_token_scaled_advantage = (
            model_logps - model_logps.detach()
        ).exp() * advantages[:, :, None]

        if self._config.loss_config.tis_imp_ratio_cap > 0:
            tis_imp_ratio = torch.exp(model_logps - vllm_logps)
            tis_imp_ratio = torch.clamp(
                tis_imp_ratio, max=self._config.loss_config.tis_imp_ratio_cap
            )
            per_token_scaled_advantage = per_token_scaled_advantage * tis_imp_ratio

        if ref_logps is not None:
            ref_logps = ref_logps.view(batch_size, num_rollouts, -1)

            # kl penalty
            kl = (ref_logps - model_logps).exp() - (ref_logps - model_logps) - 1.0
            per_token_loss = (
                per_token_scaled_advantage - self._config.loss_config.beta * kl
            )
        else:
            per_token_loss = per_token_scaled_advantage

        target_mask = target_batch.target_mask.view(batch_size, num_rollouts, -1)

        total_tokens = target_mask.sum().item()

        if self._config.loss_config.length_normalization:
            per_seq_loss = (
                (per_token_loss * target_mask).sum(dim=-1) / target_mask.sum(dim=-1)
            ).mean(dim=1)
        elif self._config.loss_config.loss_token_mean:
            per_seq_loss = per_token_loss * target_mask
        else:
            per_seq_loss = ((per_token_loss * target_mask).sum(dim=-1)).mean(dim=1)

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # self._gangs.root.barrier()

        return per_seq_loss.sum(), total_tokens

    @override
    def set_step_nr(self, step_nr: int) -> None:
        self._step_nr = step_nr

    @property
    @override
    def model(self) -> Model:
        return self._model


GRPO_FINETUNE_UNIT: Final = "grpo"


@dataclass(kw_only=True)
class GrpoLossConfig:
    group_size: int = 4
    """Number of responses to sample per prompt for advantage computation.
    
    This value must match the 'n' parameter in the VLLM sampling params.
    """

    forward_group_size: int = 4
    """Maximum number of responses to process in a single forward pass.
    
    When group_size > forward_group_size, responses are processed in multiple micro-batches
    to reduce memory usage (similar to gradient accumulation). Each micro-batch processes
    forward_group_size responses and accumulates gradients until all group_size responses
    are processed.
    """

    beta: float = 0.001
    """The coefficient of regularization towards the reference model."""

    entropy_regularizer_scale: float = 0.0
    """Scale factor for entropy regularization term."""

    length_normalization: bool = True
    """If True, normalize loss by sequence length. If False, use sequence-level loss."""

    adv_std_normalization: bool = True
    """If True, normalize advantages with standard deviation."""

    log_rollouts: bool = False
    """Log sample rollouts during training/validation."""

    loss_token_mean: bool = False
    """If True, average loss over tokens. If False, sum over tokens."""

    validation_vllm_sampling_params: Dict[str, Any] = field(default_factory=lambda: {})
    """VLLM sampling params for validation. If empty, training params will be used."""

    tis_imp_ratio_cap: float = 2.0
    """Maximum cap for the truncated importance sampling ratio. If <= 0, no cap is applied."""


@dataclass(kw_only=True)
class GrpoFinetuneConfig:
    """Configuration for Generalized Reward-Paired Optimization (GRPO) finetuning.

    GRPO finetuning uses a policy model to generate diverse responses, which are then
    evaluated by a reward model. The policy is trained to maximize the expected reward
    while maintaining proximity to a reference model.
    """

    loss_config: GrpoLossConfig = field(default_factory=lambda: GrpoLossConfig())
    """Configuration for GRPO loss computation, including rollout handling and regularization."""

    vllm_model_actor_name: str = "vllm_policy"
    """Name of the Ray vLLM actor used to generate policy rollouts."""

    vllm_reward_model_actor_name: str | None = None
    """Optional name of the Ray vLLM actor used as a reward model."""

    vllm_reference_model_actor_name: str | None = None
    """Optional name of the Ray vLLM actor used as a reference model."""

    reward: RewardSection = field(
        default_factory=lambda: RewardSection(name="gsm8k_verifier")
    )
    """Configuration for the reward function that evaluates generated rollouts."""

    vllm_sync: VllmSyncSection = field(default_factory=lambda: VllmSyncSection())


@final
class GrpoFinetuneUnitHandler(OnlineFinetuneUnitHandler):
    """
    Handles creation and configuration of GRPO fine-tuning units.
    """

    _context: RuntimeContext

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

    @override
    def create(
        self, model: Module, gangs: Gangs, recipe_config: object, vllm_actors: object
    ) -> TrainUnit[PreferenceBatch]:

        config = structure(recipe_config.criterion.config, GrpoFinetuneConfig)

        validate(config)
        log.info(f"GRPO loss config:\n{config}")

        reference_model = vllm_actors[config.vllm_reference_model_actor_name]
        if reference_model and config.vllm_sync.sync_ref_model_every_n_steps != -1:
            if reference_model and reference_model.update_process_groups is None:
                raise ValueError(
                    f"Reference model actor must have update process group if we sync weights"
                )

        vllm_model = vllm_actors[config.vllm_model_actor_name]
        if gangs.dp.rank == 0:
            if vllm_model.sampling_params.n < config.loss_config.group_size:
                log.info("Setting model sampling n to GRPO group size")
                vllm_model.sampling_params.n = config.loss_config.group_size

        vllm_reward_model = vllm_actors.get(config.vllm_reward_model_actor_name, None)
        reward_registry = self._context.get_registry(VLLMOutputRewardHandler)
        reward_name = config.reward.name
        reward_handler = reward_registry.get(reward_name)
        reward = reward_handler.create(
            reward_model=vllm_reward_model,
            reward_name=reward_name,
            reward_config=config.reward.config,
            gangs=gangs,
            context=self._context,
        )

        # sync models here before we start training
        if config.vllm_sync.sync_model_every_n_steps > 0:
            maybe_sync_model(gangs, model, vllm_model, -1, -1, force_sync=True)
        if config.vllm_sync.sync_ref_model_every_n_steps > 0:
            maybe_sync_model(gangs, model, reference_model, -1, -1, force_sync=True)

        log.info("GRPO setup complete.")

        return GrpoFinetuneUnit(
            model, reference_model, vllm_model, vllm_actors, reward, gangs, config
        )

    @property
    @override
    def name(self) -> str:
        return GRPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return GrpoFinetuneConfig
