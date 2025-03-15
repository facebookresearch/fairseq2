# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, List, cast, final

import torch.nn as nn
import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module
from torcheval.metrics import Mean
from typing_extensions import override
from fairseq2.recipes.model import Model

from fairseq2.context import RuntimeContext
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.gang import Gang, Gangs
from fairseq2.logging import log
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import (
    SequenceBatch,
    SequenceModelOutput,
    as_auto_regressive_input,
)
from fairseq2.data import CollateOptionsOverride, Collater, SequenceData
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.nn.data_parallel._fsdp import (
    fsdp_summon_full_parameters as fsdp_summon_full_parameters,
)
from fairseq2.recipes.common import setup_reference_model
from fairseq2.recipes.config import (
    ReferenceModelSection,
    TrainerSection,
    get_config_section,
)
from fairseq2.recipes.lm._preference_finetune._common import (
    POCriterionSection,
    POFinetuneMetricBag,
    _gather_lprobs_avg,
)
from fairseq2.recipes.lm._online_finetune._common import OnlineCriterionSection
from fairseq2.recipes.lm._online_finetune._handler import OnlineFinetuneUnitHandler
from fairseq2.recipes.trainer import TrainUnit
from fairseq2.typing import DataType
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate
from fairseq2.recipes.common._distributed import broadcast_model

from vllm import SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

import ray
from fairseq2.recipes.lm._online_finetune._rewards import (
    VLLMOutputRewardHandler,
    RewardSection,
    VLLMOutputReward,
)
from fairseq2.recipes.lm._online_finetune._remote_vllm import (
    VllmConfig,
    RemoteVllmModelHandler,
    RemoteVllmModel,
)

from fairseq2.recipes.lm._online_finetune._common import (
    collate_with_target_mask,
    copy_state,
    find_first_value,
    generate_rollouts,
)


@final
class OnlineDpoFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _reference_model: Module | None
    _beta: float
    _nll_scale: float
    _metric_bag: OnlineDpoFinetuneMetricBag
    _length_normalization: bool
    _model_update_group: PyNcclCommunicator
    _sync_vllm_model_every_n_steps: int
    _sync_ref_model_every_n_steps: int
    _reward: VLLMOutputReward

    def __init__(
        self,
        model: Module,
        reference_model: Module | None,
        vllm_model: RemoteVllmModel,
        reward,
        gangs: Gangs,
        beta: float = 0.1,
        nll_scale: float = 1.0,
        length_normalization: bool = False,
        sync_vllm_model_every_n_steps: int = 1,
        sync_ref_model_every_n_step: int = -1,
    ) -> None:
        super().__init__()
        self._model = model
        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale
        self._length_normalization = length_normalization
        self._vllm_model = vllm_model
        self._gangs = gangs
        self._sync_vllm_model_every_n_steps = sync_vllm_model_every_n_steps
        self._sync_ref_model_every_n_steps = sync_ref_model_every_n_step
        self._reward = reward
        self._metric_bag = OnlineDpoFinetuneMetricBag(gangs.dp)

    def maybe_sync_models(self):

        if (
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
            with self._model.summon_full_parameters():
                if self._gangs.root.rank == 0:
                    # syncing with ref model
                    copy_state(self._model.module, self._reference_model.module)
                self._gangs.root.barrier()
                broadcast_model(self._reference_model, self._gangs)

    @override
    def __call__(self, prompt_batch: PromptBatch) -> tuple[Tensor, int]:

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # self._gangs.root.barrier()

        self.maybe_sync_models()

        rollouts = generate_rollouts(
            prompt_batch.prompts, dp_gang=self._gangs.dp, vllm_model=self._vllm_model
        )

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

        if self._reference_model is not None:
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
        else:
            raise RuntimeError(
                "Reference model is not initialized and data batch does not provide reference score, but at least one must exist."
            )

        if self._length_normalization:
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

        self._metric_bag.update_batch_metrics(batch.chosen)

        avg_reward = torch.tensor(reward_output["rewards"]).float().mean()
        self._metric_bag.update_avg_reward(avg_reward)

        loss = (
            dpo_loss
            + self._nll_scale
            * nll_loss
            * chosen_target_batch.batch_size
            / chosen_target_batch.num_target_elements()
        )  # normalization applied locally per-rank

        loss = loss * loss_zeroer  # zero loss if entire batch was dummy batch

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*4, 24*4), reverse=True)

        # self._gangs.root.barrier()

        return loss, chosen_target_batch.batch_size

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
        logp_ratio_chosen = self._beta * (chosen_logps - ref_chosen_logps)
        logp_ratio_rejected = self._beta * (rejected_logps - ref_rejected_logps)
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
    avg_zeroed_loss: Mean

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("dpo_loss", Mean(device=gang.device), persistent=False)
        self.register_metric(
            "num_dummy_batches", Mean(device=gang.device), persistent=False
        )
        self.register_metric("avg_reward", Mean(device=gang.device), persistent=False)
        self.register_metric(
            "avg_zeroed_loss", Mean(device=gang.device), persistent=False
        )

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
    def update_avg_zeroed_loss(self, avg_zeroed_loss):
        self.avg_zeroed_loss.update(avg_zeroed_loss, weight=1)


ONLINE_DPO_FINETUNE_UNIT: Final = "online_dpo"


@dataclass(kw_only=True)
class OnlineDpoFinetuneConfig:
    reference_model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(name="fs2_llama3_1_8b_instruct")
    )
    """
    The reference model. If ``None``, the recipe expects to get reference
    log-probabilities for chosen and rejected targets as float values in the
    data example (fields `reference_score_rejected` and  `reference_score_chosen`).
    """

    reference_dtype: DataType = torch.bfloat16
    """The data type of the reference model."""

    # Loss
    beta: float = 0.1
    """The coefficient of regularization towards the reference model."""

    nll_scale: float = 0.0
    """The coefficient of NLL loss added to the DPO loss."""

    length_normalization: bool = False
    """Use length normalized DPO, which uses the average log probability of a sequence as the implicit reward."""

    vllm_model: VllmConfig = field(
        default_factory=lambda: VllmConfig(init_update_process_group=True)
    )

    # reward: RewardSection = field(
    #     default_factory=lambda: RewardSection(name="gsm8k_verifier")
    # )
    reward: RewardSection = field(
        default_factory=lambda: RewardSection(name="skywork_verifier")
    )

    vllm_reward_model: VllmConfig = field(
        default_factory=lambda: VllmConfig(init_update_process_group=False)
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
        self, model: Module, gangs: Gangs, recipe_config: object
    ) -> TrainUnit[PreferenceBatch]:
        criterion_section = get_config_section(
            recipe_config, "criterion", OnlineCriterionSection
        )

        vllm_model = RemoteVllmModelHandler().create(
            gangs=gangs, unit_config=config, configs_name="vllm_model"
        )
        vllm_reward_model = RemoteVllmModelHandler().create(
            gangs=gangs, unit_config=config, configs_name="vllm_reward_model"
        )

        config = structure(criterion_section.config, OnlineDpoFinetuneConfig)

        validate(config)

        reward_registry = self._context.get_registry(VLLMOutputRewardHandler)
        reward_handler = reward_registry.get(config.reward.name)
        reward = reward_handler.create(
            recipe_config=recipe_config, vllm_model=vllm_reward_model, gangs=gangs
        )

        if config.reference_model is not None:
            log.info("Setting up DPO with reference model.")

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

            log.info("DPO setup complete.")
        else:
            reference_model = None

        # if gangs.dp.rank == 0:
        # ray.init(address=f"ray://{config.ray_cluster_ip_address}:10001", namespace="vllm_workers")
        # actor_name = "vllm_model"
        # vllm_model, model_update_group = setup_vllm(actor_name, config.vllm_init_checkpoint_dir, config.vllm_init_tokenizer, config.vllm_tensor_parallel_size, gangs.dp.device)

        gangs.root.barrier()

        # if gangs.dp.rank != 0:
        #     # vllm_model = ray.get_actor(actor_name)
        #     vllm_model = None
        #     model_update_group = None

        # gangs.root.barrier()

        # if gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # gangs.root.barrier()

        return OnlineDpoFinetuneUnit(
            model,
            reference_model,
            vllm_model,
            reward,
            gangs,
            config.beta,
            config.nll_scale,
            config.length_normalization,
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
