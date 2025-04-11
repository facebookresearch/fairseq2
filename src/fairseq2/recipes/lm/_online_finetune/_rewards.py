# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List

import torch
from transformers import AutoTokenizer
from typing_extensions import override
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gangs
from fairseq2.recipes.config import get_config_section
from fairseq2.recipes.lm._online_finetune._common import (
    GRPOBatch,
    collate_with_target_mask,
    find_first_value,
    generate_rewards,
    prepare_grpo_batch,
    prepare_preference_batch_random_pair,
)
from fairseq2.recipes.lm._online_finetune._math_utils import (
    last_boxed_only_string,
    remove_boxed,
)
from fairseq2.recipes.model import Model
from fairseq2.recipes.trainer import TrainUnit
import numpy as np
from fairseq2.logging import log


@dataclass(kw_only=True)
class RewardModelConfig:
    answer_key: str = "answer"
    prompt_key: str = "prompt_raw"


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
    def create(self, reward_model, reward_config, gangs):
        return GSM8kVerifier(
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            gangs=gangs,
        )

    @property
    @override
    def name(self):
        return "gsm8k_verifier"

    @property
    @override
    def config_kls(self):
        return None


class GSM8kVerifier(VLLMOutputReward):
    def __init__(self, answer_key, prompt_key, gangs):
        self.answer_re = re.compile(
            r"#### (\-?[0-9\.\,]+)"
        )  # regexp from original gsm8k to extract formatted answer
        self.invalid_answer = "[invalid]"
        self._gangs = gangs
        self.answer_key = answer_key
        self.prompt_key = prompt_key

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
        self,
        vllm_outputs: List[RequestOutput],
        prompt_batch: PromptBatch,
    ):
        batch_text = []
        batch_tokens = []
        batch_rewards = []

        reference_answers = prompt_batch.meta_info.get(self.answer_key)

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

        reward_output = self.process_rollouts(rollouts, prompt_batch)

        batch, is_bad_batch = prepare_preference_batch_random_pair(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        return batch, is_bad_batch, reward_output

    def prepare_grpo_batch(self, prompt_batch: PromptBatch, rollouts):

        reward_output = self.process_rollouts(rollouts, prompt_batch)

        batch = prepare_grpo_batch(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        return batch, reward_output


class NuminaMathVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_config, gangs):
        return NuminaMathVerifier(
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            gangs=gangs,
        )

    @property
    @override
    def name(self):
        return "numinamath_verifier"

    @property
    @override
    def config_kls(self):
        return None


class NuminaMathVerifier(GSM8kVerifier):
    def __init__(self, answer_key, prompt_key, gangs):
        self.invalid_answer = "[invalid]"
        self._gangs = gangs
        self.answer_key = answer_key
        self.prompt_key = prompt_key

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
    def create(self, reward_model, reward_config, gangs):
        return SkyworkVerifier(
            gangs,
            reward_model,
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
        )

    @property
    @override
    def name(self):
        return "skywork_verifier"

    @property
    @override
    def config_kls(self):
        return None


class SkyworkVerifier(VLLMOutputReward):
    def __init__(self, gangs, reward_model, answer_key, prompt_key):
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self._gangs = gangs
        self.reward_model = reward_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
        )

    def wrap_text(self, prompt_text, rollout_text):
        wrapped_text = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": rollout_text},
        ]
        chat_str = self.tokenizer.apply_chat_template(wrapped_text, tokenize=False)
        chat_str = chat_str.replace("<|begin_of_text|>", "")

        return chat_str

    @override
    def process_rollouts(
        self, vllm_outputs: List[RequestOutput], prompt_batch: PromptBatch
    ):
        vllm_inputs = []
        batch_text = []
        batch_tokens = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        text_prompts = prompt_batch.meta_info.get(self.prompt_key)
        for i, (i_batch_request_output, prompt_text) in enumerate(
            zip(vllm_outputs, text_prompts)
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

        batch_rewards = generate_rewards(
            vllm_inputs, dp_gang=self._gangs.dp, vllm_model=self.reward_model
        )

        # reshape batch_rewards to [Batch, Rollouts]
        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts
        batch_rewards = [batch_rewards[i * R : (i + 1) * R] for i in range(B)]

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def get_divpo_indices(self, rewards, rollouts, p=0.10):
        cumulative_logprobs_norm = []
        for rollout_idx in range(len(rollouts[0].outputs)):
            logprobs = self.extract_logprobs(rollouts[0].outputs[rollout_idx].logprobs)
            cumulative_logprob_norm = sum(logprobs) / len(logprobs)
            cumulative_logprobs_norm.append(cumulative_logprob_norm)

        assert len(rewards) == len(
            cumulative_logprobs_norm
        ), "Rewards and logprobs must have the same length"

        # Convert the list to a numpy array
        max_val = max(rewards)
        min_val = min(rewards)

        diff = max_val - min_val
        thresh_offset = diff * p
        top_thresh = max_val - thresh_offset
        bot_thresh = min_val + thresh_offset

        chosen_set = [idx for idx, val in enumerate(rewards) if val >= top_thresh]
        rejected_set = [idx for idx, val in enumerate(rewards) if val <= bot_thresh]

        # Debugging output
        # log.info(f"rewards: {rewards}")
        # log.info(f"top_thresh: {top_thresh}, bot_thresh: {bot_thresh}")
        # log.info(f"chosen_set: {chosen_set}, rejected_set: {rejected_set}")

        max_reward_idx = min(chosen_set, key=lambda i: cumulative_logprobs_norm[i])
        min_reward_idx = max(rejected_set, key=lambda i: cumulative_logprobs_norm[i])

        return max_reward_idx, min_reward_idx

    def extract_logprobs(self, data):
        logprobs = []
        for item in data:
            for key, logprob in item.items():
                logprobs.append(logprob.logprob)
        return logprobs

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts, divpo_p=0
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

            # if self._gangs.root.rank == 0:
            #     breakpoint()

            if divpo_p > 0:
                chosen_rollout_position, rejected_rollout_position = (
                    self.get_divpo_indices(i_batch_rewards, rollouts, divpo_p)
                )
            else:
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


class AtheneVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_config, gangs):
        return AtheneVerifier(
            gangs,
            reward_model,
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
        )

    @property
    @override
    def name(self):
        return "athene_verifier"

    @property
    @override
    def config_kls(self):
        return None


class AtheneVerifier(VLLMOutputReward):
    def __init__(self, gangs, reward_model, answer_key, prompt_key):
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self._gangs = gangs
        self.reward_model = reward_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/checkpoint/ram/shared/Athene-RM-8B_tmp/"
        )

    def wrap_text(self, prompt_text, rollout_text):
        wrapped_text = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": rollout_text},
        ]
        chat_str = self.tokenizer.apply_chat_template(wrapped_text, tokenize=False)
        # chat_str = chat_str.replace("<|begin_of_text|>", "")
        chat_str += "<|reserved_special_token_1|>"

        return chat_str

    @override
    def process_rollouts(
        self, vllm_outputs: List[RequestOutput], prompt_batch: PromptBatch
    ):
        vllm_inputs = []
        batch_text = []
        batch_tokens = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        text_prompts = prompt_batch.meta_info.get(self.prompt_key)
        for i, (i_batch_request_output, prompt_text) in enumerate(
            zip(vllm_outputs, text_prompts)
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

        batch_rewards = generate_rewards(
            vllm_inputs, dp_gang=self._gangs.dp, vllm_model=self.reward_model
        )

        # reshape batch_rewards to [Batch, Rollouts]
        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts
        batch_rewards = [batch_rewards[i * R : (i + 1) * R] for i in range(B)]

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def get_divpo_indices(self, rewards, rollouts, p=0.10):
        cumulative_logprobs_norm = []
        for rollout_idx in range(len(rollouts[0].outputs)):
            logprobs = self.extract_logprobs(rollouts[0].outputs[rollout_idx].logprobs)
            cumulative_logprob_norm = sum(logprobs) / len(logprobs)
            cumulative_logprobs_norm.append(cumulative_logprob_norm)

        assert len(rewards) == len(
            cumulative_logprobs_norm
        ), "Rewards and logprobs must have the same length"

        # Convert the list to a numpy array
        max_val = max(rewards)
        min_val = min(rewards)

        diff = max_val - min_val
        thresh_offset = diff * p
        top_thresh = max_val - thresh_offset
        bot_thresh = min_val + thresh_offset

        chosen_set = [idx for idx, val in enumerate(rewards) if val >= top_thresh]
        rejected_set = [idx for idx, val in enumerate(rewards) if val <= bot_thresh]

        max_reward_idx = min(chosen_set, key=lambda i: cumulative_logprobs_norm[i])
        min_reward_idx = max(rejected_set, key=lambda i: cumulative_logprobs_norm[i])

        return max_reward_idx, min_reward_idx

    def extract_logprobs(self, data):
        logprobs = []
        for item in data:
            for key, logprob in item.items():
                logprobs.append(logprob.logprob)
        return logprobs

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts, divpo_p=0
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

            if divpo_p > 0:
                chosen_rollout_position, rejected_rollout_position = (
                    self.get_divpo_indices(i_batch_rewards, rollouts, divpo_p)
                )
            else:
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
