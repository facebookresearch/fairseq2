# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, List, cast, final

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
    SequenceData
)
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
from fairseq2.recipes.lm._online_finetune._common import setup_vllm, OnlineCriterionSection, stateless_init_process_group
from fairseq2.recipes.lm._preference_finetune._handler import POFinetuneUnitHandler
from fairseq2.recipes.trainer import TrainUnit
from fairseq2.typing import DataType
from fairseq2.utils.structured import structure
from fairseq2.utils.validation import validate

from fairseq2.recipes.lm import DpoFinetuneUnit

from vllm import SamplingParams, RequestOutput, CompletionOutput
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

import ray
from fairseq2.nn.padding import pad_seqs, get_seqs_and_padding_mask

from fairseq2.utils.env import get_rank
import os
import re

def gsm8k_correctness_verifier(vllm_output: RequestOutput, reference_answer: List[str]):
    # verifier to match predicted answer with gsm8k format with the reference

    # utils from gsm8k paper to extract a correct answer and match it to the prompt
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"

    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS

    batch_text = []
    batch_tokens = []
    batch_rewards = []

    for i, i_batch_request_output in enumerate(vllm_output):
        rollouts_text = []
        rollouts_tokens = []
        i_reference_answer = reference_answer[i]
        rollouts_rewards = []
        for rollout_output in i_batch_request_output.outputs:
            rollouts_text.append(rollout_output.text)
            rollouts_tokens.append(rollout_output.token_ids)
            predicted_answer = extract_answer(rollout_output.text)
            predicted_reward = 1 if predicted_answer == i_reference_answer else 0
            rollouts_rewards.append(predicted_reward)
        batch_text.append(rollouts_text)
        batch_tokens.append(rollouts_tokens)
        batch_rewards.append(rollouts_rewards)

    return {
        "text": batch_text,
        "tokens": batch_tokens,
        "rewards": batch_rewards
    }

def collate_with_target_mask(list_of_tensors, prompt_lens, pad_value=0, device="cpu"):
    # list_of_tensors contain prompt+rollout tokens, we use prompt_len to define the target loss mask here
    to_collate = []
    for seq, prompt_len in zip(list_of_tensors, prompt_lens):
        target_loss_mask = torch.arange(len(seq)) >= prompt_len
        to_collate.append({
            "seqs": seq,
            "target_loss_mask": target_loss_mask
        })

    # if len(to_collate) == 0:
    #     from pudb.remote import set_trace
    #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

    target_mask_collate_opts = [
            CollateOptionsOverride("target_loss_mask", pad_value=False),
        ]
    collater = Collater(pad_value=pad_value, pad_to_multiple=1, overrides=target_mask_collate_opts)

    seq_data = cast(SequenceData, collater(to_collate))

    seqs, padding_mask = get_seqs_and_padding_mask(
                seq_data["seqs"], device
            )

    batch = SequenceBatch(seqs=seqs, padding_mask=padding_mask, target_mask=seq_data["target_loss_mask"]["seqs"].to(device))

    return batch

@final
class OnlineDpoFinetuneUnit(TrainUnit[SequenceBatch]):
    """Represents the language model DPO-finetuning unit with online generations. Paper: https://arxiv.org/abs/2305.18290."""

    _reference_model: Module | None
    _beta: float
    _nll_scale: float
    _metric_bag: OnlineDpoFinetuneMetricBag
    _length_normalization: bool
    _model_update_group: PyNcclCommunicator
    _sync_model_every_n_steps: int

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
        sync_model_every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        self._model = model
        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale
        self._length_normalization = length_normalization
        self.vllm_model = vllm_model
        self.update_pg = update_pg
        self._gangs = gangs
        self._sync_model_every_n_steps = sync_model_every_n_steps

        self._metric_bag = OnlineDpoFinetuneMetricBag(gangs.dp)

    def generate_responses(self, prompts):
        sampling_params = SamplingParams(temperature=1.0)
        outputs = ray.get(self.vllm_model.generate.remote(prompts, sampling_params))
        return outputs

    def sync_weights(self):
        for name, p in self._model.module.named_parameters():
            name = name.replace("._checkpoint_wrapped_module", "")
            # print(f'sync call {name}')
            handle = self.vllm_model.collective_rpc.remote("update_weight",
                                            args=(name, p.dtype, p.shape))
            self.update_pg.broadcast(p, src=0, stream=torch.cuda.current_stream())
            ray.get(handle)

    def debatch(self, prompt_batch: SequenceBatch):
        seqs = prompt_batch.example['indices']['seqs'].tolist()
        lens = prompt_batch.example['indices']['seq_lens']

        prompt_list = [s[:l] for s,l in zip(seqs, lens)]

        return prompt_list

    def rollout_from_model(self, prompt_list, sampling_params=None):
        if sampling_params is None:
            sampling_params = SamplingParams(n=16, temperature=1.0, max_tokens=1024)

        outputs = ray.get(self.vllm_model.generate.remote(prompt_token_ids=prompt_list, sampling_params=sampling_params, use_tqdm=False))
        return outputs

    @override
    def __call__(self, prompt_batch: PromptBatch) -> tuple[Tensor, int]:

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # self._gangs.root.barrier()

        if self._step_nr % self._sync_model_every_n_steps == 0:
            with self._model.summon_full_parameters():
                if self._gangs.root.rank == 0:
                    # print(f'starting weight sync')
                    self.sync_weights()

                self._gangs.root.barrier()

        prompts_to_generate = [None]*self._gangs.dp.size
        if self._gangs.dp.rank == 0:
            self._gangs.dp.gather_object(prompt_batch.prompts, prompts_to_generate, 0)
        else:
            self._gangs.dp.gather_object(prompt_batch.prompts, None, 0)

        if self._gangs.dp.rank == 0:
            rank_batch_sizes = [len(l) for l in prompts_to_generate]
            flat_request_list = []
            for rank_prompts in prompts_to_generate:
                flat_request_list.extend(rank_prompts)
            
            rollouts = self.rollout_from_model(flat_request_list)

            rollouts_to_scatter = []
            rollouts_per_rank = [None]
            for dp_rank, rank_batch_size in zip(range(self._gangs.dp.size), rank_batch_sizes):
                rank_start = sum(rank_batch_sizes[:dp_rank])
                rank_end = rank_start + rank_batch_size
                rollouts_to_scatter.append(rollouts[rank_start:rank_end])
            self._gangs.dp.scatter_object_list(rollouts_per_rank, rollouts_to_scatter, source_rank=0)
        else:
            rollouts_per_rank = [None]
            self._gangs.dp.scatter_object_list(rollouts_per_rank, None, source_rank=0)

        # if self._gangs.root.rank == 0:
        #     from pudb.remote import set_trace
        #     set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        # self._gangs.root.barrier()

        if len(rollouts_per_rank[0]) != len(prompt_batch.meta_info["answer"]):
            from pudb.remote import set_trace
            set_trace(host="submit-0", port=6899, term_size=(80*2, 24*2), reverse=True)

        verifications = gsm8k_correctness_verifier(rollouts_per_rank[0], prompt_batch.meta_info["answer"])

        def find_first_value(lst, value):
            return next((i for i, x in enumerate(lst) if x == value), None)

        chosen_batch = []
        rejected_batch = []
        prompt_lens = []
        dummy_batch_ids = []  # keep posiitons of dummy pairs here
        loss_zeroer = 1.0
        # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
        for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(zip(verifications["rewards"],verifications["tokens"])):
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

        if len(dummy_batch_ids) == len(verifications["tokens"]):
            # entire batch does not have a valid preference pair
            # we use it as dummy batch and zero the loss in the end 
            loss_zeroer = 0.0
        else:
            # removing dummy pairs from the batch
            filter_batch = lambda batch: [item for index, item in enumerate(batch) if index not in dummy_batch_ids]
            chosen_batch = filter_batch(chosen_batch)
            rejected_batch = filter_batch(rejected_batch)
            prompt_lens = filter_batch(prompt_lens)

        prompt_lens = torch.tensor(prompt_lens)

        chosen_batch = [torch.tensor(sequence, device=self._gangs.dp.device) for sequence in chosen_batch]
        chosen_batch = collate_with_target_mask(chosen_batch, prompt_lens, device=self._gangs.dp.device)

        rejected_batch = [torch.tensor(sequence, device=self._gangs.dp.device) for sequence in rejected_batch]
        rejected_batch = collate_with_target_mask(rejected_batch, prompt_lens, device=self._gangs.dp.device)

        batch = PreferenceBatch(chosen=chosen_batch, rejected=rejected_batch, reference_score_chosen=None, reference_score_rejected=None)

        chosen_input_batch, chosen_target_batch = as_auto_regressive_input(batch.chosen)
        rejected_input_batch, rejected_target_batch = as_auto_regressive_input(
            batch.rejected
        )
        if (
            chosen_target_batch.target_mask is None
            or rejected_target_batch.target_mask is None
        ):
            raise RuntimeError("target_mask attributes must exist for DPO loss")

        chosen_output = cast(SequenceModelOutput, self._model.module(chosen_input_batch))
        rejected_output = cast(SequenceModelOutput, self._model.module(rejected_input_batch))

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
                    SequenceModelOutput, self._reference_model.module(chosen_batch)
                )
                ref_rejected_output = cast(
                    SequenceModelOutput, self._reference_model.module(rejected_batch)
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

        self._metric_bag.update_nll_loss(chosen_batch, nll_loss)

        self._metric_bag.update_sequence_lengths(batch)

        self._metric_bag.update_logps(batch, chosen_logps, rejected_logps)

        self._metric_bag.update_batch_metrics(chosen_batch)

        self._metric_bag.update_num_dummy_batches(batch, torch.tensor(len(dummy_batch_ids)))

        avg_reward = torch.tensor(verifications["rewards"]).float().mean()
        self._metric_bag.update_avg_reward(avg_reward)
        self._metric_bag.update_avg_zeroed_loss(torch.tensor(loss_zeroer))

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
        self.register_metric("num_dummy_batches", Mean(device=gang.device), persistent=False)
        self.register_metric("avg_reward", Mean(device=gang.device), persistent=False)
        self.register_metric("avg_zeroed_loss", Mean(device=gang.device), persistent=False)

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
        self.num_dummy_batches.update(num_dummy_batches / batch.chosen.batch_size, weight=batch.chosen.batch_size)

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

    ray_cluster_ip_address: str = None

    vllm_init_checkpoint_dir: str = "/checkpoint/ram/kulikov/gsm8k_8b_sft_debug/checkpoints/step_20"

    vllm_init_tokenizer: str = "/datasets/pretrained-llms/Llama-3.1-8B-Instruct/"

    vllm_tensor_parallel_size: int = 4

    sync_model_every_n_steps: int = 5

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

        # test_out = test_generation(vllm_model, gangs.dp.rank)

        # print(f"rank:{gangs.dp.rank}, out: {test_out}")

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
            config.sync_model_every_n_steps,
        )
    
    @property
    @override
    def name(self) -> str:
        return ONLINE_DPO_FINETUNE_UNIT

    @property
    @override
    def config_kls(self) -> type[object]:
        return OnlineDpoFinetuneConfig
