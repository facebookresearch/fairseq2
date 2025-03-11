# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing_extensions import override
from typing import Any, List

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gangs
from fairseq2.recipes.model import Model
from fairseq2.recipes.trainer import TrainUnit
from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput
from fairseq2.recipes.lm._online_finetune._common import collate_with_target_mask, find_first_value
import re
from fairseq2.recipes.config import (
    get_config_section,
)


@dataclass(kw_only=True)
class RewardSection:
    name: str
    

class VLLMOutputRewardHandler(ABC):
    @abstractmethod
    def create(
        self, reward_model: Any, gangs: Gangs, recipe_config: object
    ) -> VLLMOutputReward: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class VLLMOutputReward(ABC):
    @abstractmethod
    def process_rollouts(self, vllm_outputs: List[RequestOutput]): ...

    def prepare_preference_batch(self, prompt_batch: PromptBatch, rollouts): ...


class GSM8kVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, recipe_config, gangs):
        return GSM8kVerifier(gangs)
    
    @property
    @override
    def name(self):
        return "gsm8k_verifier"
    
    @property
    @override
    def config_kls(self):
        return None


class GSM8kVerifier(VLLMOutputReward):
    def __init__(self, gangs):
        self.answer_re = re.compile(r"#### (\-?[0-9\.\,]+)")  # regexp from original gsm8k to extract formatted answer
        self.invalid_answer = "[invalid]"
        self._gangs = gangs

    def extract_answer(self, completion: str):
        match = self.answer_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.invalid_answer

    @override
    def process_rollouts(self, vllm_outputs: List[RequestOutput], reference_answers: List[str]):
        batch_text = []
        batch_tokens = []
        batch_rewards = []

        for i, i_batch_request_output in enumerate(vllm_outputs):
            rollouts_text = []
            rollouts_tokens = []
            i_reference_answer = reference_answers[i]
            rollouts_rewards = []
            for rollout_output in i_batch_request_output.outputs:
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)
                predicted_answer = self.extract_answer(rollout_output.text)
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

    def prepare_preference_batch(self, prompt_batch: PromptBatch, rollouts) -> PreferenceBatch:

        reward_output = self.process_rollouts(rollouts, prompt_batch.meta_info["answer"])

        chosen_batch = []
        rejected_batch = []
        prompt_lens = []
        dummy_batch_ids = []  # keep posiitons of dummy pairs here
        
        # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
        for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(zip(reward_output["rewards"],reward_output["tokens"])):
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

        filter_batch = lambda batch: [item for index, item in enumerate(batch) if index not in dummy_batch_ids]

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

        chosen_batch = [torch.tensor(sequence, device=self._gangs.dp.device) for sequence in chosen_batch]
        chosen_batch = collate_with_target_mask(chosen_batch, prompt_lens, device=self._gangs.dp.device)

        rejected_batch = [torch.tensor(sequence, device=self._gangs.dp.device) for sequence in rejected_batch]
        rejected_batch = collate_with_target_mask(rejected_batch, prompt_lens, device=self._gangs.dp.device)

        batch = PreferenceBatch(chosen=chosen_batch, rejected=rejected_batch, reference_score_chosen=None, reference_score_rejected=None)

        return batch, is_bad_batch, reward_output

    def prepare_grpo_batch(self, prompt_batch: PromptBatch, rollouts):
        raise NotImplementedError()