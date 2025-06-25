# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Dict, Final, List, cast, final, Any, Union

import ray
import torch
import torch.distributed
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from fairseq2.metrics import Mean, MetricBag
from typing_extensions import override
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from fairseq2.nn._batch_layout import BatchLayout
from fairseq2.recipes.lm._online_finetune._common import (
    compute_token_level_entropy,
    log_rollouts,
    get_rollout_lengths,
)

from fairseq2.context import RuntimeContext
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.models.clm import CausalLM
from fairseq2.datasets import (
    LengthBatching,
    SequenceBatch,
    StaticBatching,
    SyncMode,
)
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.recipes.common import setup_reference_model
from fairseq2.recipes.common._distributed import broadcast_model
from fairseq2.recipes.config import (
    ReferenceModelSection,
    TrainerSection,
)
from fairseq2.recipes.lm._online_finetune._common import (
    GRPOBatch,
    OnlineCriterionSection,
    collate_with_target_mask,
    combine_prompts_responses_for_scoring,
    convert_vllm_output_to_ref_score,
    copy_state,
    generate_rollouts,
    prepare_grpo_batch,
    StatefulRolloutBag,
    update_num_dummy_batches,
    update_avg_loss_zeroer,
    update_avg_reward,
    update_dpo_loss,
    update_avg_reward_len_norm,
    update_avg_rollout_length,
    update_batch_metrics,
    update_logit_entropy,
    update_grpo_loss
)
from fairseq2.recipes.lm._online_finetune._handler import OnlineFinetuneUnitHandler
from fairseq2.recipes.lm._online_finetune._remote_model import (
    RemoteVllmModel,
    RemoteHFModel,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    RewardSection,
    VLLMOutputReward,
    VLLMOutputRewardHandler,
)
from fairseq2.recipes import Model, TrainUnit
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

from fairseq2.models.llama._hg import _convert_parameter


@final
class GrpoFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _step_nr: int
    _reference_model: Module | RemoteVllmModel | None
    _vllm_model: RemoteVllmModel
    _vllm_actors: Dict[str, Union[RemoteVllmModel, RemoteHFModel]]
    _loss_config: GrpoLossConfig
    _model_update_group: PyNcclCommunicator
    _sync_vllm_model_every_n_steps: int
    _sync_ref_model_every_n_steps: int
    _reward: VLLMOutputReward
    _display_name: str
    _reference_offload: bool
    _rollout_bag: StatefulRolloutBag

    def __init__(
        self,
        model: Module,
        reference_model: Module | RemoteVllmModel,
        reference_offload: bool,
        vllm_model: RemoteVllmModel,
        vllm_actors: List[Union[RemoteVllmModel, RemoteHFModel]],
        reward,
        gangs: Gangs,
        loss_config: GrpoLossConfig,
        sync_vllm_model_every_n_steps: int = 1,
        sync_ref_model_every_n_step: int = -1,
    ) -> None:
        super().__init__()
        self._model = model
        self._reference_model = reference_model
        self._loss_config = loss_config
        self._vllm_actors = vllm_actors
        self._vllm_model = vllm_model
        self._gangs = gangs
        self._sync_vllm_model_every_n_steps = sync_vllm_model_every_n_steps
        self._sync_ref_model_every_n_steps = sync_ref_model_every_n_step
        self._reward = reward
        self._reference_offload = reference_offload
        self._rollout_bag = StatefulRolloutBag()

        self._display_name = "GRPO"

    @property
    @override
    def display_name(self) -> str | None:
        return self._display_name

    def maybe_sync_models(self, force_sync_vllm=False):
        if force_sync_vllm or (
            self._sync_vllm_model_every_n_steps > 0
            and self._step_nr % self._sync_vllm_model_every_n_steps == 0
        ):
            with self._model.summon_full_parameters():
                if self._gangs.dp.rank == 0:
                    self._vllm_model.sync_weights_with_vllm(model=self._model, converter=_convert_parameter)
                self._gangs.root.barrier()

        if hasattr(self, "_step_nr") and (
            self._sync_ref_model_every_n_steps > 0
            and self._step_nr % self._sync_ref_model_every_n_steps == 0
        ):

            if self._reference_offload:
                with self._model.summon_full_parameters():
                    if self._gangs.dp.rank == 0:
                        self._reference_model.sync_weights_with_vllm(model=self._model, converter=_convert_parameter)
                    self._gangs.root.barrier()
            else:
                with self._model.summon_full_parameters():
                    if self._gangs.root.rank == 0:
                        # syncing with ref model
                        copy_state(self._model.module, self._reference_model.module)
                    self._gangs.root.barrier()
                    broadcast_model(self._reference_model, self._gangs)

    def validate_reward(self, prompt_batch: PromptBatch, metric_bag) -> tuple[Tensor, int]:
        if self._gangs.dp.rank == 0:
            policy_sampling_params = copy(self._vllm_model.sampling_params)
            policy_sampling_params.n = 1
            for k, v in self._loss_config.validation_vllm_sampling_params.items():
                policy_sampling_params.__setattr__(k, v)
        else:
            policy_sampling_params = None
        rollouts = generate_rollouts(
            prompt_batch.prompts,
            dp_gang=self._gangs.dp,
            vllm_model=self._vllm_model,
            sampling_params=policy_sampling_params,
        )
        if self._loss_config.log_rollouts:
            log_rollouts(prompt_batch, rollouts, "Valid")
        reward_output = self._reward.process_rollouts(rollouts, prompt_batch)
        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()

        rollout_lengths = get_rollout_lengths(rollouts)
        avg_rollout_length = torch.tensor(rollout_lengths).float().mean()
        avg_reward_len_norm = avg_reward / avg_rollout_length

        update_avg_rollout_length(metric_bag, avg_rollout_length)
        update_avg_reward_len_norm(metric_bag, avg_reward_len_norm)

        update_avg_reward(metric_bag, avg_reward)
        update_batch_metrics(metric_bag, prompt_batch, train=False)
        # returning dummy loss since trainer expects it
        return torch.tensor(0.0, device=self._gangs.dp.device), prompt_batch.batch_size

    def compute_reference_logps(self, seqs: Tensor, layout: BatchLayout, prompt_lengths: list[int]):
        
        seqs_to_score = seqs.tolist()
        if layout.padded:
            padding_mask = layout.position_indices >= 0  # True when non-pad
            seqs_to_score = [
                seq[:l]
                for seq, l in zip(
                    seqs_to_score, padding_mask.sum(-1).tolist()
                )
            ]

        scored_responses = generate_rollouts(
            seqs_to_score, dp_gang=self._gangs.dp, vllm_model=self._reference_model
        )
        ref_logps = convert_vllm_output_to_ref_score(scored_responses, self._gangs)
        ref_logps = collate_with_target_mask(
            ref_logps, prompt_lengths, device=self._gangs.dp.device
        ).seqs

        # if self._gangs.root.rank == 0:
        #     import ipdb; ipdb.set_trace()
        # self._gangs.root.barrier()

        return ref_logps

    @override
    def __call__(self, prompt_batch: PromptBatch, metric_bag: MetricBag) -> tuple[Tensor, int]:

        if not self.model.module.training:
            # we are in valid mode, only compute reward and return
            dummy_loss, batch_size = self.validate_reward(prompt_batch, metric_bag=metric_bag)
            return dummy_loss, batch_size

        self._rollout_bag.maybe_reset_bag(self._step_nr)

        if len(self._rollout_bag) == 0:

            self.maybe_sync_models()

            rollouts = generate_rollouts(
                prompt_batch.prompts,
                dp_gang=self._gangs.dp,
                vllm_model=self._vllm_model,
            )
            if self._loss_config.log_rollouts:
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
                self._loss_config.forward_group_size
            ),
        )

        # grpo_batch, reward_output = self._reward.prepare_grpo_batch(prompt_batch, rollouts)  # loss_zeroer is used when entire batch has no valid prefrence pair

        grpo_input_batch, grpo_target_batch = grpo_batch.prompt_rollouts.as_auto_regressive()
        grpo_input_batch_seqs, grpo_input_batch_seqs_layout = grpo_input_batch.as_input()

        grpo_model_logits = self._model.module(grpo_input_batch_seqs, grpo_input_batch_seqs_layout)

        logps = self._gather_lprobs(grpo_model_logits, grpo_target_batch)

        tgt_logit_entropy = compute_token_level_entropy(
            grpo_model_logits, grpo_target_batch.target_mask
        )  # [Batch x Rollouts, 1]

        max_entropy_regularizer = (
            -tgt_logit_entropy.sum() * self._loss_config.entropy_regularizer_scale
        )
        update_logit_entropy(metric_bag, tgt_logit_entropy)

        prompt_rollout_seqs, prompt_rollout_layout = grpo_batch.prompt_rollouts.as_input()
        ref_logps = self.compute_reference_logps(prompt_rollout_seqs, prompt_rollout_layout, grpo_batch.prompt_lengths)

        _grpo_objective = self._compute_grpo_objective(
            logps, ref_logps, grpo_batch.rewards, grpo_target_batch
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

        # self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        update_batch_metrics(metric_bag,
            grpo_batch.prompt_rollouts, train=True
        )  # TODO fix, now logs only the last prompt from the batch

        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()
        update_avg_reward(metric_bag, avg_reward)

        loss = grpo_loss

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*4, 24*4), reverse=True)

        # self._gangs.root.barrier()

        return loss, prompt_batch.batch_size

    def _gather_lprobs(
        self, logits: Tensor, target: SequenceBatch
    ) -> Tensor:
        assert target.target_mask is not None
        logprobs = torch.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(
            -1
        )  # [Batch, 1]

        return per_token_logps

    def _compute_grpo_objective(
        self,
        logps,
        ref_logps,
        advantages: Tensor,  # outcome based only for now
        target_batch: SequenceBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:

        batch_size = advantages.size(0)
        num_rollouts = advantages.size(1)
        logps = logps.view(batch_size, num_rollouts, -1)
        ref_logps = ref_logps.view(batch_size, num_rollouts, -1)

        # kl penalty
        kl = (ref_logps - logps).exp() - (ref_logps - logps) - 1.0

        per_token_scaled_advantage = (logps - logps.detach()).exp() * advantages[
            :, :, None
        ]
        # per_token_scaled_advantage = logps * advantages[:,:,None]

        per_token_loss = per_token_scaled_advantage - self._loss_config.beta * kl

        target_mask = target_batch.target_mask.view(batch_size, num_rollouts, -1)

        if self._loss_config.length_normalization:
            per_seq_loss = (
                (per_token_loss * target_mask).sum(dim=-1) / target_mask.sum(dim=-1)
            ).mean(dim=1)
        else:
            per_seq_loss = ((per_token_loss * target_mask).sum(dim=-1)).mean(dim=1)

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # self._gangs.root.barrier()

        return per_seq_loss.sum()

    @override
    def set_step_nr(self, step_nr: int) -> None:
        self._step_nr = step_nr

    @property
    @override
    def model(self) -> Model:
        return self._model


# class GrpoFinetuneMetricBag(SequenceMetricBag):
#     """Holds the metrics of a DPO preference finetuning task."""

#     # rollout_logps: Mean
#     rollout_lengths: Mean
#     grpo_loss: Mean
#     logit_entropy: Mean
#     avg_reward: Mean

#     def __init__(self, gang: Gang) -> None:
#         super().__init__(gang)

#         # self.register_metric("rollout_logps", Mean(device=gang.device), persistent=False)
#         self.register_metric(
#             "rollout_lengths", Mean(device=gang.device), persistent=False
#         )
#         self.register_metric("grpo_loss", Mean(device=gang.device), persistent=False)
#         self.register_metric("avg_reward", Mean(device=gang.device), persistent=False)
#         self.register_metric(
#             "avg_rollout_length", Mean(device=gang.device), persistent=False
#         )
#         self.register_metric(
#             "avg_reward_len_norm", Mean(device=gang.device), persistent=False
#         )
#         self.register_metric(
#             "logit_entropy", Mean(device=gang.device), persistent=False
#         )

#     @torch.inference_mode()
#     def update_logit_entropy(self, logit_entropy: Tensor):
#         # logit_entropy is expected to contain token-level entropy for every sequence in the current batch
#         batch_size = logit_entropy.size(0)
#         self.logit_entropy.update(logit_entropy.sum() / batch_size, weight=batch_size)

#     @torch.inference_mode()
#     def update_grpo_loss(self, batch: PromptBatch, loss: Tensor) -> None:
#         """Update the GRPO loss metric.

#         :param batch:
#             The batch processed by the model.
#         :param loss:
#             The GRPO loss of ``batch``.
#         """
#         self.grpo_loss.update(loss / batch.batch_size, weight=batch.batch_size)

#     @torch.inference_mode()
#     def update_rollout_lengths(
#         self,
#         rollout_lengths: Tensor,
#     ) -> None:
#         """Update the Chosen Sequence Length and Rejected Sequence Length metrics.

#         :param batch:
#             The batch processed by the model.
#         """
#         self.rollout_lengths.update(
#             rollout_lengths.mean(),
#             weight=rollout_lengths.size(0),
#         )

#     @torch.inference_mode()
#     def update_avg_reward(self, avg_reward):
#         self.avg_reward.update(avg_reward, weight=1)

#     @torch.inference_mode()
#     def update_avg_rollout_length(self, avg_rollout_length):
#         self.avg_rollout_length.update(avg_rollout_length, weight=1)

#     @torch.inference_mode()
#     def update_avg_reward_len_norm(self, avg_reward_len_norm):
#         self.avg_reward_len_norm.update(avg_reward_len_norm, weight=1)

#     @torch.inference_mode()
#     def update_batch_metrics(self, batch: PreferenceBatch):
#         num_examples = batch.batch_size
#         self.num_examples.update(num_examples)
#         if self._train:
#             assert self.total_num_examples is not None
#             self.total_num_examples.update(num_examples)


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

    log_rollouts: bool = False
    """Log sample rollouts during training/validation."""

    validation_vllm_sampling_params: Dict[str, Any] = field(default_factory=lambda: {})
    """VLLM sampling params for validation. If empty, training params will be used."""


@dataclass(kw_only=True)
class GrpoFinetuneConfig:
    """Configuration for Generalized Reward-Paired Optimization (GRPO) finetuning.

    GRPO finetuning uses a policy model to generate diverse responses, which are then
    evaluated by a reward model. The policy is trained to maximize the expected reward
    while maintaining proximity to a reference model.
    """

    reference_model: ReferenceModelSection | str = field(
        default_factory=lambda: ReferenceModelSection(name="fs2_llama3_1_8b_instruct")
    )
    """
    The reference model for KL regularization. If set to string, reference
    log-probabilities are obtained from the specified vLLM actor.
    """

    loss_config: GrpoLossConfig = field(default_factory=lambda: GrpoLossConfig())
    """Configuration for GRPO loss computation, including rollout handling and regularization."""

    reference_dtype: torch.dtype = torch.bfloat16
    """The data type of the reference model when loaded locally."""

    ray_policy_actor_name: str = "vllm_policy"
    """Name of the Ray vLLM actor used to generate policy rollouts."""

    vllm_reward_model_name: str | None = None
    """Optional name of the Ray vLLM actor used as a reward model."""

    reward: RewardSection = field(
        default_factory=lambda: RewardSection(name="gsm8k_verifier")
    )
    """Configuration for the reward function that evaluates generated rollouts."""

    sync_ref_model_every_n_steps: int = -1
    """How often to sync the reference model with the policy. -1 disables syncing."""

    sync_vllm_model_every_n_steps: int = -1
    """How often to sync the vLLM model with the policy. -1 disables syncing."""


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

        config = structure(
            recipe_config.criterion.config, GrpoFinetuneConfig
        )

        validate(config)
        log.info(f"GRPO loss config:\n{config}")

        if isinstance(config.reference_model, ReferenceModelSection):
            log.info("Setting up GRPO with reference model.")

            trainer_section = structure(
                recipe_config.trainer, TrainerSection
            )

            reference_model = setup_reference_model(
                CausalLM,
                self._context,
                config.reference_model.name,
                gangs,
                config.reference_dtype,
                mp=False,
                torch_compile=trainer_section.torch_compile,
            )

            freeze_parameters(reference_model.module)
            reference_offload = False

        elif isinstance(config.reference_model, str):
            reference_model = vllm_actors[config.reference_model]
            reference_offload = True
            if config.sync_ref_model_every_n_steps != -1:
                if reference_model and reference_model.update_process_groups is None:
                    raise ValueError(
                        f"Reference model actor must have update process group if we sync weights"
                    )

        else:
            raise ValueError(f"reference model {config.reference_model} not supported")

        gangs.root.barrier()

        vllm_model = vllm_actors[config.ray_policy_actor_name]
        if gangs.dp.rank == 0:
            if vllm_model.sampling_params.n != config.loss_config.group_size:
                raise RuntimeError(
                    "GRPO policy sampling n must match loss config group_size"
                )

        vllm_reward_model = vllm_actors.get(config.vllm_reward_model_name, None)
        reward_registry = self._context.get_registry(VLLMOutputRewardHandler)
        reward_handler = reward_registry.get(config.reward.name)
        reward = reward_handler.create(
            reward_model=vllm_reward_model,
            reward_config=config.reward.config,
            gangs=gangs,
        )

        log.info("GRPO setup complete.")

        return GrpoFinetuneUnit(
            model,
            reference_model,
            reference_offload,
            vllm_model,
            vllm_actors,
            reward,
            gangs,
            config.loss_config,
            config.sync_vllm_model_every_n_steps,
            config.sync_ref_model_every_n_steps,
        )

    @property
    @override
    def name(self) -> str:
        return GRPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return GrpoFinetuneConfig
