# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Dict, Final, List, cast, final, Any

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
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
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
    get_config_section,
)
from fairseq2.recipes.lm._online_finetune._common import (
    OnlineCriterionSection,
    collate_with_target_mask,
    convert_vllm_output_to_ref_score,
    copy_state,
    find_first_value,
    generate_rollouts,
    log_rollouts,
)
from fairseq2.recipes.lm._online_finetune._handler import OnlineFinetuneUnitHandler
from fairseq2.recipes.lm._online_finetune._remote_vllm import (
    RemoteVllmModel,
    RemoteVllmModelHandler,
)
from fairseq2.recipes.lm._online_finetune._rewards import (
    RewardSection,
    VLLMOutputReward,
    VLLMOutputRewardHandler,
)
from fairseq2.recipes.lm._preference_finetune._common import (
    POCriterionSection,
    POFinetuneMetricBag,
    _gather_lprobs_avg,
)
from fairseq2.recipes.lm._online_finetune._common import compute_token_level_entropy
from fairseq2.recipes.model import Model
from fairseq2.recipes.trainer import TrainUnit
from fairseq2.typing import DataType
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate


@final
class OnlineDpoFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _reference_model: Module | RemoteVllmModel | None
    _vllm_model: RemoteVllmModel
    _vllm_actors: Dict[str, RemoteVllmModel]
    _metric_bag: OnlineDpoFinetuneMetricBag
    _loss_config: DpoLossConfig
    _model_update_group: PyNcclCommunicator
    _sync_vllm_model_every_n_steps: int
    _sync_ref_model_every_n_steps: int
    _display_name: str
    _reward: VLLMOutputReward
    _reference_offload: bool

    def __init__(
        self,
        model: Module,
        reference_model: Module | RemoteVllmModel,
        reference_offload: bool,
        vllm_model: RemoteVllmModel,
        vllm_actors: List[RemoteVllmModel],
        reward,
        gangs: Gangs,
        loss_config: DpoLossConfig,
        sync_vllm_model_every_n_steps: int = 1,
        sync_ref_model_every_n_step: int = -1,
    ) -> None:
        super().__init__()
        self._model = model
        self._reference_model = reference_model
        self._reference_offload = reference_offload
        self._vllm_actors = vllm_actors
        self._loss_config = loss_config
        self._vllm_model = vllm_model
        self._gangs = gangs
        self._sync_vllm_model_every_n_steps = sync_vllm_model_every_n_steps
        self._sync_ref_model_every_n_steps = sync_ref_model_every_n_step
        self._reward = reward
        self._metric_bag = OnlineDpoFinetuneMetricBag(gangs.dp)

        self._display_name = "online_dpo"

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
                if self._gangs.root.rank == 0:
                    self._vllm_model.sync_weights_with_vllm(train_model=self._model)
                self._gangs.root.barrier()

        if (
            self._sync_ref_model_every_n_steps > 0
            and self._step_nr % self._sync_ref_model_every_n_steps == 0
        ):

            if self._reference_offload:
                with self._model.summon_full_parameters():
                    if self._gangs.root.rank == 0:
                        self._reference_model.sync_weights_with_vllm(
                            train_model=self._model
                        )
                    self._gangs.root.barrier()
            else:
                with self._model.summon_full_parameters():
                    if self._gangs.root.rank == 0:
                        # syncing with ref model
                        copy_state(self._model.module, self._reference_model.module)
                    self._gangs.root.barrier()
                    broadcast_model(self._reference_model, self._gangs)

    def validate_reward(self, prompt_batch: PromptBatch) -> tuple[Tensor, int]:
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

        self._metric_bag.update_avg_reward(avg_reward)
        self._metric_bag.update_batch_metrics(prompt_batch)
        # returning dummy loss since trainer expects it
        return torch.tensor(0.0, device=self._gangs.dp.device), prompt_batch.batch_size

    def compute_reference_logps(self, seq_batch: SequenceBatch):
        seqs_to_score = seq_batch.seqs.tolist()
        if seq_batch.padding_mask:
            prompt_lengths = (
                (~seq_batch.target_mask)
                .logical_and(seq_batch.padding_mask.materialize())
                .sum(dim=-1)
                .cpu()
            )  # extracting actual prompt lengths
            seqs_to_score = [
                seq[:l]
                for seq, l in zip(
                    seqs_to_score, seq_batch.padding_mask.seq_lens.tolist()
                )
            ]
        else:
            prompt_lengths = (~seq_batch.target_mask).sum(dim=-1).cpu()

        scored_responses = generate_rollouts(
            seqs_to_score, dp_gang=self._gangs.dp, vllm_model=self._reference_model
        )
        ref_logps = convert_vllm_output_to_ref_score(scored_responses, self._gangs)
        ref_logps = collate_with_target_mask(
            ref_logps, prompt_lengths, device=self._gangs.dp.device
        ).seqs

        return ref_logps

    def get_all_rollouts_entropy(self, rollouts):
        all_entropy = []
        for rollout_idx in range(len(rollouts[0].outputs)):
            logprobs = rollouts[0].outputs[rollout_idx].logprobs
            logprobs = [next(iter(x.values())).logprob for x in logprobs]
            entropy = sum(logprobs) / len(logprobs)
            all_entropy.append(entropy)
        logit_entropy = torch.tensor(all_entropy, device=self._gangs.dp.device)
        return logit_entropy

    @override
    def __call__(self, prompt_batch: PromptBatch) -> tuple[Tensor, int]:

        if not self.model.module.training:
            # we are in valid mode, only compute reward and return
            dummy_loss, batch_size = self.validate_reward(prompt_batch)
            return dummy_loss, batch_size

        self.maybe_sync_models()

        rollouts = generate_rollouts(
            prompt_batch.prompts, dp_gang=self._gangs.dp, vllm_model=self._vllm_model
        )
        if self._loss_config.log_rollouts:
            log_rollouts(prompt_batch, rollouts, "Train")

        batch: PreferenceBatch
        batch, is_bad_batch, reward_output = self._reward.prepare_preference_batch(
            prompt_batch, rollouts
        )  # loss_zeroer is used when entire batch has no valid prefrence pair
        if is_bad_batch:
            loss_zeroer = 0.0
        else:
            loss_zeroer = 1.0

        # below is the usual DPO code
        chosen_input_batch, chosen_target_batch = as_auto_regressive_input(batch.chosen)
        rejected_input_batch, rejected_target_batch = as_auto_regressive_input(
            batch.rejected
        )
        if (
            chosen_target_batch.target_mask is None
            or rejected_target_batch.target_mask is None
        ):
            raise RuntimeError("target_mask attributes must exist for DPO loss")

        chosen_output = cast(
            SequenceModelOutput, self._model.module(chosen_input_batch)
        )
        rejected_output = cast(
            SequenceModelOutput, self._model.module(rejected_input_batch)
        )

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # self._gangs.root.barrier()

        chosen_logps, average_chosen_logps = _gather_lprobs_avg(
            chosen_output, chosen_target_batch
        )
        rejected_logps, average_rejected_logps = _gather_lprobs_avg(
            rejected_output, rejected_target_batch
        )
        chosen_tgt_logit_entropy = compute_token_level_entropy(
            chosen_output.logits, chosen_target_batch.target_mask
        )  # [Batch x Rollouts, 1]
        rejected_tgt_logit_entropy = compute_token_level_entropy(
            rejected_output.logits, rejected_target_batch.target_mask
        )  # [Batch x Rollouts, 1]

        # entropy for all N rollouts
        logit_entropy = self.get_all_rollouts_entropy(rollouts)

        max_entropy_regularizer = (
            -chosen_tgt_logit_entropy.sum()
            * self._loss_config.entropy_regularizer_scale
        )

        self.metric_bag.update_logit_entropy(logit_entropy)
        self.metric_bag.update_chosen_logit_entropy(chosen_tgt_logit_entropy)
        self.metric_bag.update_rejected_logit_entropy(rejected_tgt_logit_entropy)

        if self._reference_offload:
            token_ref_chosen_logps = self.compute_reference_logps(batch.chosen)
            token_ref_rejected_logps = self.compute_reference_logps(batch.rejected)

            ref_average_chosen_logps = token_ref_chosen_logps.mean(dim=-1)
            ref_average_rejected_logps = token_ref_rejected_logps.mean(dim=-1)

            ref_chosen_logps = token_ref_chosen_logps.sum(dim=-1)
            ref_rejected_logps = token_ref_rejected_logps.sum(dim=-1)

        else:
            with torch.no_grad():
                ref_chosen_output = cast(
                    SequenceModelOutput, self._reference_model.module(batch.chosen)
                )
                ref_rejected_output = cast(
                    SequenceModelOutput, self._reference_model.module(batch.rejected)
                )
                ref_chosen_logps, ref_average_chosen_logps = _gather_lprobs_avg(
                    ref_chosen_output, chosen_target_batch
                )
                ref_rejected_logps, ref_average_rejected_logps = _gather_lprobs_avg(
                    ref_rejected_output, rejected_target_batch
                )

        if self._loss_config.length_normalization:
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

        nll_loss = chosen_output.compute_loss(
            chosen_target_batch.seqs, loss_mask=chosen_target_batch.target_mask
        )

        self._metric_bag.update_dpo_loss(batch, dpo_loss)

        self._metric_bag.update_nll_loss(batch.chosen, nll_loss)

        self._metric_bag.update_sequence_lengths(batch)

        self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        self._metric_bag.update_avg_loss_zeroer(torch.tensor(loss_zeroer))

        self._metric_bag.update_batch_metrics(batch.chosen)

        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()
        self._metric_bag.update_avg_reward(avg_reward)

        loss = (
            dpo_loss
            + self._loss_config.nll_scale
            * nll_loss
            * chosen_target_batch.batch_size
            / chosen_target_batch.num_target_elements()
            + max_entropy_regularizer
        )  # nll normalization applied locally per-rank

        loss = loss * loss_zeroer  # zero loss if entire batch was dummy batch

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*4, 24*4), reverse=True)

        # self._gangs.root.barrier()

        return loss, prompt_batch.batch_size

    def _gather_lprobs(
        self, output: SequenceModelOutput, target: SequenceBatch
    ) -> tuple[Tensor, Tensor]:
        assert target.target_mask is not None
        logprobs = torch.log_softmax(output.logits, dim=-1)
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
        logp_ratio_chosen = self._loss_config.beta * (chosen_logps - ref_chosen_logps)
        logp_ratio_rejected = self._loss_config.beta * (
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

    @property
    @override
    def metric_bag(self) -> OnlineDpoFinetuneMetricBag:
        return self._metric_bag


class OnlineDpoFinetuneMetricBag(POFinetuneMetricBag):
    """Holds the metrics of a DPO preference finetuning task."""

    dpo_loss: Mean
    num_dummy_batches: Mean
    avg_reward: Mean
    avg_loss_zeroer: Mean
    chosen_logit_entropy: Mean
    rejected_logit_entropy: Mean
    logit_entropy: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("dpo_loss", Mean(device=gang.device), persistent=False)
        self.register_metric(
            "num_dummy_batches", Mean(device=gang.device), persistent=False
        )
        self.register_metric("avg_reward", Mean(device=gang.device), persistent=False)
        self.register_metric(
            "avg_loss_zeroer", Mean(device=gang.device), persistent=False
        )
        self.register_metric(
            "logit_entropy", Mean(device=gang.device), persistent=False
        )

    @torch.inference_mode()
    def update_chosen_logit_entropy(self, logit_entropy: Tensor):
        # logit_entropy for chosen sequences
        batch_size = logit_entropy.size(0)
        self.chosen_logit_entropy.update(
            logit_entropy.sum() / batch_size, weight=batch_size
        )

    @torch.inference_mode()
    def update_rejected_logit_entropy(self, logit_entropy: Tensor):
        # logit_entropy for rejected sequences
        batch_size = logit_entropy.size(0)
        self.rejected_logit_entropy.update(
            logit_entropy.sum() / batch_size, weight=batch_size
        )

    @torch.inference_mode()
    def update_logit_entropy(self, logit_entropy: Tensor):
        # logit_entropy is expected to contain token-level entropy for every sequence in the current batch
        batch_size = logit_entropy.size(0)
        self.logit_entropy.update(logit_entropy.sum() / batch_size, weight=batch_size)

    @torch.inference_mode()
    def update_dpo_loss(self, batch: PreferenceBatch, loss: Tensor) -> None:
        """Update the DPO loss metric.

        :param batch:
            The batch processed by the model.
        :param loss:
            The DPO loss of ``batch``.
        """
        self.dpo_loss.update(
            loss / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )

    @torch.inference_mode()
    def update_num_dummy_batches(self, batch: PreferenceBatch, num_dummy_batches: int):
        self.num_dummy_batches.update(
            num_dummy_batches / batch.chosen.batch_size, weight=batch.chosen.batch_size
        )

    @torch.inference_mode()
    def update_avg_reward(self, avg_reward):
        self.avg_reward.update(avg_reward, weight=1)

    @torch.inference_mode()
    def update_avg_loss_zeroer(self, avg_loss_zeroer):
        self.avg_loss_zeroer.update(avg_loss_zeroer, weight=1)

    @torch.inference_mode()
    def update_batch_metrics(self, batch: PreferenceBatch):
        num_examples = batch.batch_size
        self.num_examples.update(num_examples)
        if self._train:
            assert self.total_num_examples is not None
            self.total_num_examples.update(num_examples)


ONLINE_DPO_FINETUNE_UNIT: Final = "online_dpo"


@dataclass(kw_only=True)
class DpoLossConfig:
    # Loss
    beta: float = 0.1
    """The coefficient of regularization towards the reference model."""

    nll_scale: float = 0.0
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
    reference_model: ReferenceModelSection | str = field(
        default_factory=lambda: ReferenceModelSection(name="fs2_llama3_1_8b_instruct")
    )
    """
    The reference model. If set to string, the recipe expects to get reference
    log-probabilities for rollouts using vllm actor.
    """

    reference_dtype: DataType = torch.bfloat16
    """The data type of the reference model."""

    loss_config: DpoLossConfig = field(default_factory=lambda: DpoLossConfig())

    ray_policy_actor_name: str = "vllm_policy"
    vllm_reward_model_name: str = None

    reward: RewardSection = field(
        default_factory=lambda: RewardSection(name="gsm8k_verifier")
    )

    sync_ref_model_every_n_steps: int = -1
    sync_vllm_model_every_n_steps: int = -1


@final
class OnlineDpoFinetuneUnitHandler(OnlineFinetuneUnitHandler):
    _context: RuntimeContext

    def __init__(self, context: RuntimeContext) -> None:
        self._context = context

    @override
    def create(
        self, model: Module, gangs: Gangs, recipe_config: object, vllm_actors: object
    ) -> TrainUnit[PreferenceBatch]:
        criterion_section = get_config_section(
            recipe_config, "criterion", OnlineCriterionSection
        )

        config = structure(criterion_section.config, OnlineDpoFinetuneConfig)

        validate(config)

        if isinstance(config.reference_model, ReferenceModelSection):
            log.info("Setting up GRPO with reference model.")

            trainer_section = get_config_section(
                recipe_config, "trainer", TrainerSection
            )

            reference_model = setup_reference_model(
                DecoderModel,
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
                if reference_model and reference_model.update_process_group is None:
                    raise ValueError(
                        f"Reference model actor must have update process group if we sync weights"
                    )

        else:
            raise ValueError(f"reference model {config.reference_model} not supported")

        gangs.root.barrier()

        vllm_model = vllm_actors[config.ray_policy_actor_name]

        vllm_reward_model = vllm_actors.get(config.vllm_reward_model_name, None)
        reward_registry = self._context.get_registry(VLLMOutputRewardHandler)
        reward_handler = reward_registry.get(config.reward.name)
        reward = reward_handler.create(
            reward_model=vllm_reward_model,
            reward_config=config.reward.config,
            gangs=gangs,
        )

        return OnlineDpoFinetuneUnit(
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
        return ONLINE_DPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return OnlineDpoFinetuneConfig
