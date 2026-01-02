# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Union, cast, final

import ray
import torch
import torch.distributed
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean
from typing_extensions import override
from vllm import SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from fairseq2.context import RuntimeContext
from fairseq2.data import CollateOptionsOverride, Collater, SequenceData
from fairseq2.datasets import LengthBatching, SequenceBatch, StaticBatching, SyncMode
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.metrics import Mean, MetricBag
from fairseq2.models.clm import CausalLM
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.nn.utils.module import freeze_parameters

# from fairseq2.recipes.model import Model
from fairseq2.recipes import Model, TrainUnit
from fairseq2.recipes.common import setup_reference_model
from fairseq2.recipes.common._distributed import broadcast_model
from fairseq2.recipes.config import ReferenceModelSection, TrainerSection
from fairseq2.recipes.lm._instruction_finetune import update_nll_loss
from fairseq2.recipes.lm._online_finetune._common import (
    VllmSyncSection,
    compute_reference_logps,
    compute_token_level_entropy,
    generate_rollouts,
    get_rollout_lengths,
    log_rollouts,
    update_avg_loss_zeroer,
    update_avg_reward,
    update_avg_reward_len_norm,
    update_avg_rollout_length,
    update_batch_metrics,
    update_dpo_loss,
    update_grpo_batch_metrics,
    compute_reference_logps,
    collate_with_target_mask,
    update_avg_loss_zeroer,
    strip_think_tokens,
    update_logit_entropy,
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
from fairseq2.recipes.lm._preference_finetune._common import (
    _gather_lprobs_avg,
    update_logps_metrics,
    update_sequence_length_metrics,
)

# from fairseq2.typing import DataType
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@final
class OnlineDpoFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _step_nr: int
    _valid_step_nr: int
    _reference_model: Module | RemoteVllmModel | None
    _vllm_model: RemoteVllmModel
    _vllm_actors: Dict[str, Union[RemoteVllmModel, RemoteHFModel]]
    _config: OnlineDpoFinetuneConfig
    _model_update_group: PyNcclCommunicator
    _display_name: str
    _reward: VLLMOutputReward

    def __init__(
        self,
        model: Module,
        reference_model: Module | RemoteVllmModel,
        vllm_model: RemoteVllmModel,
        vllm_actors: List[Union[RemoteVllmModel, RemoteHFModel]],
        reward,
        gangs: Gangs,
        config: OnlineDpoFinetuneConfig,
    ) -> None:
        super().__init__()
        self._model = model
        self._reference_model = reference_model
        self._vllm_actors = vllm_actors
        self._config = config
        self._vllm_model = vllm_model
        self._gangs = gangs
        self._reward = reward

        self._display_name = "online_dpo"

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
            for (
                k,
                v,
            ) in self._config.loss_config.validation_vllm_sampling_params.items():
                policy_sampling_params.__setattr__(k, v)

            # For a pairwise RM, need to sample at least two rollouts
            policy_sampling_params.n = (
                2 if self._reward.reward_name == "generative_pairwise_verifier" else 1
            )
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

        rollouts = strip_think_tokens(rollouts)
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

    @override
    def __call__(
        self, prompt_batch: PromptBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:

        if not self.model.module.training:
            # we are in valid mode, only compute reward and return
            dummy_loss, batch_size = self.validate_reward(prompt_batch, metric_bag)
            return dummy_loss, batch_size

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
            prompt_batch.prompts, dp_gang=self._gangs.dp, vllm_model=self._vllm_model
        )
        if self._config.loss_config.log_rollouts:
            log_rollouts(prompt_batch, rollouts, "Train")

        rollouts = strip_think_tokens(rollouts)

        batch: PreferenceBatch
        batch, is_bad_batch, reward_output = self._reward.prepare_preference_batch(
            prompt_batch, rollouts
        )  # loss_zeroer is used when entire batch has no valid prefrence pair
        if is_bad_batch:
            loss_zeroer = 0.0
        else:
            loss_zeroer = 1.0

        # below is the usual DPO code
        chosen_input_batch, chosen_target_batch = batch.chosen.as_auto_regressive()
        (
            chosen_input_batch_seqs,
            chosen_input_batch_layout,
        ) = chosen_input_batch.as_input()

        (
            rejected_input_batch,
            rejected_target_batch,
        ) = batch.rejected.as_auto_regressive()
        (
            rejected_input_batch_seqs,
            rejected_input_batch_layout,
        ) = rejected_input_batch.as_input()

        if (
            chosen_target_batch.target_mask is None
            or rejected_target_batch.target_mask is None
        ):
            raise RuntimeError("target_mask attributes must exist for DPO loss")

        nll_loss, chosen_logits = self._model.module(
            chosen_input_batch_seqs,
            chosen_input_batch_layout,
            targets=chosen_target_batch.seqs,
            target_mask=chosen_target_batch.target_mask,
            return_logits=True,
        )

        rejected_logits = self._model.module(
            rejected_input_batch_seqs, rejected_input_batch_layout
        )

        chosen_logps, average_chosen_logps = _gather_lprobs_avg(
            chosen_logits, chosen_target_batch
        )
        rejected_logps, average_rejected_logps = _gather_lprobs_avg(
            rejected_logits, rejected_target_batch
        )

        tgt_logit_entropy = compute_token_level_entropy(
            chosen_logits, chosen_target_batch.target_mask
        )  # [Batch x Rollouts, 1]

        max_entropy_regularizer = (
            -tgt_logit_entropy.sum()
            * self._config.loss_config.entropy_regularizer_scale
        )
        update_logit_entropy(metric_bag, tgt_logit_entropy)

        # reward_output["prompt_lengths"] excludes lengths that correspond to examples that we filter out due to no preference signal.
        prompt_lengths = (
            prompt_batch.prompt_lengths
            if is_bad_batch
            else reward_output["prompt_lengths"]
        )

        token_ref_chosen_logps = compute_reference_logps(
            self._gangs,
            self._reference_model,
            chosen_input_batch_seqs,
            chosen_input_batch_layout,
            prompt_lengths,
        )

        token_ref_rejected_logps = compute_reference_logps(
            self._gangs,
            self._reference_model,
            rejected_input_batch_seqs,
            rejected_input_batch_layout,
            prompt_lengths,
        )

        ref_average_chosen_logps = token_ref_chosen_logps.mean(dim=-1)
        ref_average_rejected_logps = token_ref_rejected_logps.mean(dim=-1)

        ref_chosen_logps = token_ref_chosen_logps.sum(dim=-1)
        ref_rejected_logps = token_ref_rejected_logps.sum(dim=-1)

        if self._config.loss_config.length_normalization:
            _, _, dpo_loss = self._compute_dpo_loss(
                average_chosen_logps,
                ref_average_chosen_logps,
                average_rejected_logps,
                ref_average_rejected_logps,
            )
        else:
            _, _, dpo_loss = self._compute_dpo_loss(
                chosen_logps, ref_chosen_logps, rejected_logps, ref_rejected_logps
            )

        update_dpo_loss(metric_bag, dpo_loss, batch.chosen.batch_size)

        update_nll_loss(metric_bag, nll_loss, batch.chosen.num_target_elements)

        update_sequence_length_metrics(metric_bag, batch)

        update_logps_metrics(metric_bag, batch, chosen_logps, rejected_logps)

        update_avg_loss_zeroer(metric_bag, torch.tensor(loss_zeroer))

        update_batch_metrics(metric_bag, batch.chosen, train=True)

        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()
        update_avg_reward(metric_bag, avg_reward)

        if self._config.loss_config.nll_length_normalization:
            nll_loss = (
                nll_loss
                * chosen_target_batch.batch_size
                / chosen_target_batch.num_target_elements
            )

        loss = (
            dpo_loss
            + self._config.loss_config.nll_scale * nll_loss
            + max_entropy_regularizer
        )  # nll normalization applied locally per-rank

        loss = loss * loss_zeroer  # zero loss if entire batch was dummy batch

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*4, 24*4), reverse=True)

        # self._gangs.root.barrier()

        return loss, prompt_batch.batch_size

    def _gather_lprobs(
        self, logits: Tensor, target: SequenceBatch
    ) -> tuple[Tensor, Tensor]:
        assert target.target_mask is not None
        logprobs = torch.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(logprobs, -1, target.seqs.unsqueeze(-1)).squeeze(
            -1
        )
        total_logps = (per_token_logps * target.target_mask).sum(dim=-1)  # [Batch, 1]
        assert target.target_mask is not None
        average_logps = total_logps / target.target_mask.sum(-1)

        return total_logps, average_logps

    def _compute_dpo_loss(
        self,
        chosen_logps: Tensor,
        ref_chosen_logps: Tensor,
        rejected_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logp_ratio_chosen = self._config.loss_config.beta * (
            chosen_logps - ref_chosen_logps
        )
        logp_ratio_rejected = self._config.loss_config.beta * (
            rejected_logps - ref_rejected_logps
        )
        dpo_loss = -torch.nn.functional.logsigmoid(
            logp_ratio_chosen - logp_ratio_rejected
        )
        return logp_ratio_chosen, logp_ratio_rejected, dpo_loss.sum()

    @override
    def set_step_nr(self, step_nr: int) -> None:
        self._step_nr = step_nr

    @property
    @override
    def model(self) -> Model:
        return self._model


ONLINE_DPO_FINETUNE_UNIT: Final = "online_dpo"


@dataclass(kw_only=True)
class DpoLossConfig:
    # Loss
    beta: float = 0.1
    """The coefficient of regularization towards the reference model."""

    nll_scale: float = 0.0
    nll_length_normalization: bool = True
    """The coefficient of NLL loss added to the DPO loss."""

    length_normalization: bool = False
    """Use length normalized DPO, which uses the average log probability of a sequence as the implicit reward."""

    entropy_regularizer_scale: float = 0.0

    log_rollouts: bool = False
    """Log rollouts during training/validation"""

    validation_vllm_sampling_params: Dict[str, Any] = field(default_factory=lambda: {})
    """VLLM sampling params for validation. If not set, the same params as training will be used."""


@dataclass(kw_only=True)
class OnlineDpoFinetuneConfig:
    vllm_model_actor_name: str = "vllm_policy"
    """Name of the Ray vLLM actor used to generate policy rollouts."""

    vllm_reward_model_actor_name: str | None = None
    """Optional name of the Ray vLLM actor used as a reward model."""

    vllm_reference_model_actor_name: str | None = None
    """Optional name of the Ray vLLM actor used as a reference model."""

    loss_config: DpoLossConfig = field(default_factory=lambda: DpoLossConfig())

    reward: RewardSection = field(
        default_factory=lambda: RewardSection(name="gsm8k_verifier")
    )

    vllm_sync: VllmSyncSection = field(default_factory=lambda: VllmSyncSection())


@final
class OnlineDpoFinetuneUnitHandler(OnlineFinetuneUnitHandler):
    _context: RuntimeContext

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

    @override
    def create(
        self, model: Module, gangs: Gangs, recipe_config: object, vllm_actors: object
    ) -> TrainUnit[PreferenceBatch]:
        config = structure(recipe_config.criterion.config, OnlineDpoFinetuneConfig)

        validate(config)

        reference_model = vllm_actors[config.vllm_reference_model_actor_name]
        if config.vllm_sync.sync_ref_model_every_n_steps != -1:
            if reference_model and reference_model.update_process_groups is None:
                raise ValueError(
                    f"Reference model actor must have update process group if we sync weights"
                )

        vllm_model = vllm_actors[config.vllm_model_actor_name]

        vllm_reward_model = vllm_actors.get(config.vllm_reward_model_actor_name, None)
        reward_registry = self._context.get_registry(VLLMOutputRewardHandler)
        reward_handler = reward_registry.get(config.reward.name)
        reward_name = config.reward.name
        reward = reward_handler.create(
            reward_model=vllm_reward_model,
            reward_name=reward_name,
            reward_config=config.reward.config,
            gangs=gangs,
            context=self._context,
        )


        # TODO: decide converter as part of the model handler
        if "llama" in model.name:
            from fairseq2.models.llama._hg import _convert_parameter

            model._convert_parameter = _convert_parameter
        else:
            from fairseq2.models.qwen._hg import _convert_parameter

            model._convert_parameter = _convert_parameter

        # sync models here before we start training
        if config.vllm_sync.sync_model_every_n_steps > 0:
            maybe_sync_model(gangs, model, vllm_model, -1, -1, force_sync=True)
        if config.vllm_sync.sync_ref_model_every_n_steps > 0:
            maybe_sync_model(gangs, model, reference_model, -1, -1, force_sync=True)


        return OnlineDpoFinetuneUnit(
            model, reference_model, vllm_model, vllm_actors, reward, gangs, config
        )

    @property
    @override
    def name(self) -> str:
        return ONLINE_DPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return OnlineDpoFinetuneConfig
