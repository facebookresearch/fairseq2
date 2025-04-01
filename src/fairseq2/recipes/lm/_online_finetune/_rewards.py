# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
from typing_extensions import override
from typing import Any, List

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gangs
from fairseq2.recipes.model import Model
from fairseq2.recipes.trainer import TrainUnit
from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput
from fairseq2.recipes.lm._online_finetune._common import (
    collate_with_target_mask,
    find_first_value,
    GRPOBatch,
    prepare_preference_batch_random_pair,
    prepare_grpo_batch,
)
import re
from fairseq2.recipes.config import (
    get_config_section,
)
from fairseq2.recipes.lm._online_finetune._math_utils import (
    remove_boxed,
    last_boxed_only_string,
)
from transformers import AutoTokenizer


@dataclass(kw_only=True)
class RewardModelConfig:
    answer_key: str = "answer"


@dataclass(kw_only=True)
class RewardSection:
    name: str = "dummy"
    config: RewardModelConfig = field(default_factory=lambda: RewardModelConfig())


class VLLMOutputRewardHandler(ABC):
    @abstractmethod
    def create(
        self, reward_model: Any, gangs: Gangs, reward_config: object
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

    @abstractmethod
    def prepare_preference_batch(self, prompt_batch: PromptBatch, rollouts): ...

    @abstractmethod
    def prepare_grpo_batch(self, prompt_batch: PromptBatch, rollouts): ...


class GSM8kVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_config, gangs):
        return GSM8kVerifier(answer_key=reward_config.answer_key, gangs=gangs)

    @property
    @override
    def name(self):
        return "gsm8k_verifier"

    @property
    @override
    def config_kls(self):
        return None


class GSM8kVerifier(VLLMOutputReward):
    def __init__(self, answer_key, gangs):
        self.answer_re = re.compile(
            r"#### (\-?[0-9\.\,]+)"
        )  # regexp from original gsm8k to extract formatted answer
        self.invalid_answer = "[invalid]"
        self._gangs = gangs
        self.answer_key = answer_key

    def extract_answer(self, completion: str):
        match = self.answer_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.invalid_answer

    @override
    def process_rollouts(
        self, vllm_outputs: List[RequestOutput], reference_answers: List[str]
    ):
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

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:

        reward_output = self.process_rollouts(
            rollouts, prompt_batch.meta_info[self.answer_key]
        )

        batch, is_bad_batch = prepare_preference_batch_random_pair(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        return batch, is_bad_batch, reward_output

    def prepare_grpo_batch(self, prompt_batch: PromptBatch, rollouts):

        reward_output = self.process_rollouts(
            rollouts, prompt_batch.meta_info[self.answer_key]
        )

        batch = prepare_grpo_batch(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        return batch, reward_output


class NuminaMathVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_config, gangs):
        return NuminaMathVerifier(answer_key=reward_config.answer_key, gangs=gangs)

    @property
    @override
    def name(self):
        return "numinamath_verifier"

    @property
    @override
    def config_kls(self):
        return None


class NuminaMathVerifier(GSM8kVerifier):
    def __init__(self, answer_key, gangs):
        self.invalid_answer = "[invalid]"
        self._gangs = gangs
        self.answer_key = answer_key

    def extract_answer(self, completion: str):
        try:
            parsed_answer = remove_boxed(last_boxed_only_string(completion))
        except:
            return self.invalid_answer
        return parsed_answer


class SkyworkVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, recipe_config, vllm_model, gangs):
        return SkyworkVerifier(gangs, vllm_model)

    @property
    @override
    def name(self):
        return "skywork_verifier"

    @property
    @override
    def config_kls(self):
        return None


class SkyworkVerifier(VLLMOutputReward):
    def __init__(self, gangs, vllm_model):
        self.answer_re = re.compile(
            r"#### (\-?[0-9\.\,]+)"
        )  # regexp from original gsm8k to extract formatted answer
        self.invalid_answer = "[invalid]"
        self._gangs = gangs
        self.vllm_model = vllm_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
        )

    def extract_answer(self, completion: str):
        match = self.answer_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.invalid_answer

    def extract_text_from_llama3_wrapper(self, input_string):
        start_pattern = r"<\|start_header_id\|>user<\|end_header_id\|>"
        end_pattern = r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>"
        start_index = re.search(start_pattern, input_string).end()
        end_index = re.search(end_pattern, input_string).start()
        # Extract the text between the start and end indices
        extracted_text = input_string[start_index:end_index].strip()
        return extracted_text

    def wrap_text(self, prompt_text, rollout_text):
        wrapped_text = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": rollout_text},
        ]
        # templated_text = self.tokenizer.apply_chat_template(wrapped_text, tokenize=True)
        # tokens_prompt = TokensPrompt(prompt_token_ids=templated_text)
        chat_str = self.tokenizer.apply_chat_template(wrapped_text, tokenize=False)
        chat_str = chat_str.replace("<|begin_of_text|>", "")

        return chat_str

    @override
    def process_rollouts(
        self,
        vllm_outputs: List[RequestOutput],
        prompt_batch,
    ):
        vllm_inputs = []
        batch_text = []
        batch_tokens = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        for i, (i_batch_request_output, prompt, prompt_text) in enumerate(
            zip(vllm_outputs, prompt_batch.prompts, prompt_batch.meta_info["prompt"])
        ):

            rollouts_text = []
            rollouts_tokens = []
            for rollout_output in i_batch_request_output.outputs:
                rollout_text = rollout_output.text
                vllm_input = self.wrap_text(prompt_text, rollout_text)
                vllm_inputs.append(vllm_input)
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)

            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)

        batch_rewards = generate_rollouts(
            vllm_inputs,
            dp_gang=self._gangs.dp,
            vllm_model=self.vllm_model,
            operation="reward",
        )

        # reshape batch_rewards to [Batch, Rollouts]
        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts
        batch_rewards = [batch_rewards[i * R : (i + 1) * R] for i in range(B)]

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:

        reward_output = self.process_rollouts(rollouts, prompt_batch)

        chosen_batch = []
        rejected_batch = []
        prompt_lens = []
        dummy_batch_ids = []  # keep posiitons of dummy pairs here

        # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
        for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
            zip(reward_output["rewards"], reward_output["tokens"])
        ):
            chosen_rollout_position = i_batch_rewards.index(max(i_batch_rewards))
            rejected_rollout_position = i_batch_rewards.index(min(i_batch_rewards))

            if chosen_rollout_position == rejected_rollout_position:
                # cant form preference pair when we dont have such rollouts
                # this will be dummy batch and we zero out loss
                dummy_batch_ids.append(i_batch)

            chosen_rollout_tokens = list(i_batch_tokens[chosen_rollout_position])
            rejected_rollout_tokens = list(i_batch_tokens[rejected_rollout_position])
            prompt_tokens = prompt_batch.prompts[i_batch]

            chosen_tokens = prompt_tokens + chosen_rollout_tokens
            chosen_batch.append(chosen_tokens)

            rejected_tokens = prompt_tokens + rejected_rollout_tokens
            rejected_batch.append(rejected_tokens)

            prompt_lens.append(len(prompt_tokens))

        filter_batch = lambda batch: [
            item for index, item in enumerate(batch) if index not in dummy_batch_ids
        ]

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

        chosen_batch = [
            torch.tensor(sequence, device=self._gangs.dp.device)
            for sequence in chosen_batch
        ]
        chosen_batch = collate_with_target_mask(
            chosen_batch, prompt_lens, device=self._gangs.dp.device
        )

        rejected_batch = [
            torch.tensor(sequence, device=self._gangs.dp.device)
            for sequence in rejected_batch
        ]
        rejected_batch = collate_with_target_mask(
            rejected_batch, prompt_lens, device=self._gangs.dp.device
        )

        batch = PreferenceBatch(
            chosen=chosen_batch,
            rejected=rejected_batch,
            reference_score_chosen=None,
            reference_score_rejected=None,
        )

        return batch, is_bad_batch, reward_output

    def prepare_grpo_batch(self, prompt_batch: PromptBatch, rollouts):

        prompt_rollouts = []
        prompt_lens = []
        rewards = []

        reward_output = self.process_rollouts(rollouts, prompt_batch)

        for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
            zip(reward_output["rewards"], reward_output["tokens"])
        ):
            prompt = prompt_batch.prompts[i_batch]
            rollout_tokens = [
                torch.tensor(prompt + list(c), device=self._gangs.dp.device)
                for c in i_batch_tokens
            ]
            prompt_rollouts.extend(rollout_tokens)

            prompt_lens.extend([len(prompt)] * len(rollout_tokens))

            rewards.append(i_batch_rewards)

        prompt_rollout_batch = collate_with_target_mask(
            prompt_rollouts, prompt_lens, device=self._gangs.dp.device
        )

        rewards = torch.tensor(
            rewards, device=self._gangs.dp.device
        ).float()  # [Batch, Rollouts]
        rewards_normalized = (rewards - rewards.mean(dim=1, keepdim=True)) / (
            rewards.std(dim=1, keepdim=True) + 1e-6
        )  # small epsilon to compensate 0 std

        grpo_batch = GRPOBatch(
            prompt_rollouts=prompt_rollout_batch, rewards=rewards_normalized
        )

        return grpo_batch, reward_output
