# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, cast, final

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
from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
)
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.nn.data_parallel._fsdp import summon_fsdp as summon_fsdp
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
from fairseq2.recipes.lm._online_finetune._common import setup_vllm, OnlineCriterionSection, stateless_init_process_group
from fairseq2.recipes.lm._preference_finetune._handler import POFinetuneUnitHandler
from fairseq2.recipes.trainer import TrainUnit
from fairseq2.typing import DataType
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

from fairseq2.recipes.lm import DpoFinetuneUnit

from vllm import SamplingParams
from vllm.utils import get_open_port, get_ip
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

import ray


@final
class OnlineDpoFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _reference_model: Module | None
    _beta: float
    _nll_scale: float
    _metric_bag: OnlineDpoFinetuneMetricBag
    _length_normalization: bool
    _model_update_group: PyNcclCommunicator

    def __init__(
        self,
        model: Module,
        reference_model: Module | None,
        vllm_model,
        update_pg,
        gangs: Gangs,
        beta: float = 0.1,
        nll_scale: float = 1.0,
        length_normalization: bool = False,
    ) -> None:
        super().__init__()

        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale
        self._length_normalization = length_normalization
        self.vllm_model = vllm_model
        self.update_pg = update_pg
        self._gangs = gangs

        self._metric_bag = OnlineDpoFinetuneMetricBag(gangs.dp)

    def generate_responses(self, prompts):
        sampling_params = SamplingParams(temperature=1.0)
        outputs = ray.get(self.vllm_model.generate.remote(prompts, sampling_params))
        return outputs

    def sync_weights(self):
        print(f'print before sync')
        for name, p in self._model.named_parameters():
            name = name.replace("._checkpoint_wrapped_module", "")
            # print(f'sync call {name}')
            handle = self.vllm_model.collective_rpc.remote("update_weight",
                                            args=(name, p.dtype, p.shape))
            self.update_pg.broadcast(p, src=0, stream=torch.cuda.current_stream())
            ray.get(handle)
        print(f'print after sync')

    def debatch(self, prompt_batch: SequenceBatch):
        seqs = prompt_batch.example['indices']['seqs'].tolist()
        lens = prompt_batch.example['indices']['seq_lens']

        prompt_list = [s[:l] for s,l in zip(seqs, lens)]

        return prompt_list

    def rollout_from_model(self, prompt_list, sampling_params=None):
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=1.0, max_tokens=1024)

        outputs = ray.get(self.vllm_model.generate.remote(prompt_token_ids=prompt_list, sampling_params=sampling_params))
        return outputs    

    @override
    def __call__(self, prompt_batch: SequenceBatch) -> tuple[Tensor, int]:

        prompt_list = self.debatch(prompt_batch)

        outputs = self.rollout_from_model(prompt_list)

        self._gangs.root.barrier()

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # self._gangs.root.barrier()

        with summon_fsdp(self._model):
            if self._gangs.root.rank == 0:
                print(f'starting weight sync')
                self.sync_weights()

            self._gangs.root.barrier()

        outputs1 = self.rollout_from_model(prompt_list)

        self._gangs.root.barrier()

        if self._gangs.root.rank == 0:
            from pudb.remote import set_trace
            set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # self._gangs.root.barrier()

        self._gangs.root.barrier()

        batch = prompt_batch

        chosen_batch = batch.chosen
        chosen_input_batch, chosen_target_batch = as_auto_regressive_input(chosen_batch)
        rejected_batch = batch.rejected
        rejected_input_batch, rejected_target_batch = as_auto_regressive_input(
            rejected_batch
        )
        if (
            chosen_target_batch.target_mask is None
            or rejected_target_batch.target_mask is None
        ):
            raise RuntimeError("target_mask attributes must exist for DPO loss")

        chosen_output = cast(SequenceModelOutput, self._model(chosen_input_batch))
        rejected_output = cast(SequenceModelOutput, self._model(rejected_input_batch))

        chosen_logps, average_chosen_logps = _gather_lprobs_avg(
            chosen_output, chosen_target_batch
        )
        rejected_logps, average_rejected_logps = _gather_lprobs_avg(
            rejected_output, rejected_target_batch
        )

        if self._reference_model is not None:
            with torch.no_grad():
                ref_chosen_output = cast(
                    SequenceModelOutput, self._reference_model(chosen_batch)
                )
                ref_rejected_output = cast(
                    SequenceModelOutput, self._reference_model(rejected_batch)
                )
                ref_chosen_logps, ref_average_chosen_logps = _gather_lprobs_avg(
                    ref_chosen_output, chosen_target_batch
                )
                ref_rejected_logps, ref_average_rejected_logps = _gather_lprobs_avg(
                    ref_rejected_output, rejected_target_batch
                )
        elif (
            batch.reference_score_chosen is not None
            and batch.reference_score_rejected is not None
        ):
            # reference scores must exist in the batch if reference model is None
            ref_chosen_logps = batch.reference_score_chosen
            ref_average_chosen_logps = (
                ref_chosen_logps / chosen_target_batch.target_mask.sum(-1)
            )
            ref_rejected_logps = batch.reference_score_rejected
            ref_average_rejected_logps = (
                ref_rejected_logps / rejected_target_batch.target_mask.sum(-1)
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

        self._metric_bag.update_nll_loss(chosen_batch, nll_loss)

        self._metric_bag.update_sequence_lengths(batch)

        self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        self._metric_bag.update_batch_metrics(chosen_batch)

        loss = (
            dpo_loss
            + self._nll_scale
            * nll_loss
            * chosen_target_batch.batch_size
            / chosen_target_batch.num_target_elements()
        )  # normalization applied locally per-rank

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

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)

        self.register_metric("online_dpo_loss", Mean(device=gang.device), persistent=False)

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

    ray_cluster_ip_address: str = None

    vllm_init_checkpoint_dir: str = "/checkpoint/ram/kulikov/gsm8k_8b_sft/checkpoints/step_20"

    vllm_init_tokenizer: str = "/datasets/pretrained-llms/Llama-3.1-8B-Instruct/"

    vllm_tensor_parallel_size: int = 4

def test_generation(llm, number):
    prompt = [f"<|start_header_id|>assistant<|end_header_id|> repeat this number: {number}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"]
    sampling_params = SamplingParams(n=8, temperature=1.0)
    outputs = ray.get(llm.generate.remote(prompt, sampling_params))
    output = outputs[0].outputs[0].text
    return output


@final
class OnlineDpoFinetuneUnitHandler(POFinetuneUnitHandler):
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

        config = structure(criterion_section.config, OnlineDpoFinetuneConfig)

        validate(config)

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

        ray.init(address=f"ray://{config.ray_cluster_ip_address}:10001", namespace="vllm_workers")

        actor_name = "vllm_model"

        if gangs.dp.rank == 0:
            vllm_model, model_update_group = setup_vllm(actor_name, config.vllm_init_checkpoint_dir, config.vllm_init_tokenizer, config.vllm_tensor_parallel_size, gangs.dp.device)
    
        gangs.root.barrier()

        if gangs.dp.rank != 0:
            vllm_model = ray.get_actor(actor_name)

            model_update_group = None

        test_out = test_generation(vllm_model, gangs.dp.rank)

        print(f"rank:{gangs.dp.rank}, out: {test_out}")

        # initialize model sync process group on first dp rank

        gangs.root.barrier()

        # if gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # gangs.root.barrier()

        return OnlineDpoFinetuneUnit(
            model,
            reference_model,
            vllm_model,
            model_update_group,
            gangs,
            config.beta,
            config.nll_scale,
            config.length_normalization,
        )
    
    @property
    @override
    def name(self) -> str:
        return ONLINE_DPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return OnlineDpoFinetuneConfig
