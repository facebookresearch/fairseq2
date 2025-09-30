# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from fairseq2.context import RuntimeContext
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipes.lm._online_finetune._common import (
    _mute_output,
    collate_with_target_mask,
    generate_rewards,
    generate_rewards_generative,
    generate_rollouts,
    prepare_preference_batch_random_pair,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    JudgmentExtractorHandler,
)
from transformers import AutoTokenizer
from typing_extensions import override
from vllm import CompletionOutput, LLM, RequestOutput, SamplingParams

log._logger.setLevel(logging.DEBUG)


@dataclass(kw_only=True)
class RewardModelConfig:
    answer_key: str = "answer"
    prompt_key: str = "prompt"
    tokenizer: str | None = None
    judgment_extractor: str | None = None
    additional_fields: Dict[str, Any] | None = None


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
    def process_rollouts(self, vllm_outputs: list[RequestOutput]): ...

    @abstractmethod
    def prepare_preference_batch(self, prompt_batch: PromptBatch, rollouts): ...


class GSM8kVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        return GSM8kVerifier(
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            reward_name=reward_name,
            gangs=gangs,
            context=context,
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
    def __init__(self, answer_key, prompt_key, reward_name, gangs, context):
        self.answer_re = re.compile(
            r"#### (\-?[0-9\.\,]+)"
        )  # regexp from original gsm8k to extract formatted answer
        self.invalid_answer = "[invalid]"
        self._gangs = gangs
        self._context = context
        self.answer_key = answer_key
        self.reward_name = reward_name
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
        vllm_outputs: list[RequestOutput],
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

        batch, is_bad_batch, dummy_batch_ids = prepare_preference_batch_random_pair(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        prompt_lengths = [
            l
            for idx, l in enumerate(prompt_batch.prompt_lengths)
            if idx not in dummy_batch_ids
        ]
        reward_output["prompt_lengths"] = prompt_lengths

        return batch, is_bad_batch, reward_output


class MathVerifyHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        return MathVerifyVerifier(
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            reward_name=reward_name,
            gangs=gangs,
            context=context,
        )

    @property
    @override
    def name(self):
        return "math_verify"

    @property
    @override
    def config_kls(self):
        return None


class MathVerifyVerifier(VLLMOutputReward):
    def __init__(self, answer_key, prompt_key, reward_name, gangs, context):
        try:
            from math_verify.metric import math_metric
            from math_verify.parser import (
                ExprExtractionConfig,
                LatexExtractionConfig,
                NormalizationConfig,
            )
        except ImportError:
            raise ImportError(
                "install mathverify from https://github.com/huggingface/Math-Verify"
            )

        self._gangs = gangs
        self._context = context
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self.reward_name = reward_name

        label_normalizer = NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed="none",
            equations=False,
        )
        self.verify_func = math_metric(
            gold_extraction_target=(
                LatexExtractionConfig(normalization_config=label_normalizer),
            ),
            pred_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
            aggregation_function=max,
            precision=6,
        )

    def verify_answer(self, completion: str, answer: str):
        # here we add extra $$ to label so that LatexExtractor works as expected
        if not answer.startswith("$"):
            answer = f"${answer}$"
        try:
            with _mute_output():
                grade, extracted_answers = self.verify_func([answer], [completion])
        except:
            grade = 0
            extracted_answers = None
        reward = 1.0 if grade == 1 else 0.0

        return reward, extracted_answers

    @override
    def process_rollouts(
        self,
        vllm_outputs: list[RequestOutput],
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
                predicted_reward, predicted_answer = self.verify_answer(
                    rollout_output.text, i_reference_answer
                )
                rollouts_rewards.append(predicted_reward)
            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)
            batch_rewards.append(rollouts_rewards)

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:
        reward_output = self.process_rollouts(rollouts, prompt_batch)

        batch, is_bad_batch, dummy_batch_ids = prepare_preference_batch_random_pair(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        prompt_lengths = [
            l
            for idx, l in enumerate(prompt_batch.prompt_lengths)
            if idx not in dummy_batch_ids
        ]
        reward_output["prompt_lengths"] = prompt_lengths

        return batch, is_bad_batch, reward_output


class AtheneVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        if reward_config.tokenizer is not None:
            tokenizer = reward_config.tokenizer
        else:
            tokenizer = "Nexusflow/Athene-RM-8B"

        return AtheneVerifier(
            gangs,
            context,
            reward_model,
            reward_name=reward_name,
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            tokenizer=tokenizer,
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
    """
    A reward model verifier that processes rollouts using the Athene reward model.

    This class evaluates rollouts generated by vLLM by wrapping the prompt and rollout text into a specific format and passing it through the Athene reward model.

    Note: this relies on modified Athene-RM-8B code to ensure compatibility with vLLM.
    """

    def __init__(
        self,
        gangs,
        context,
        reward_model,
        reward_name,
        answer_key,
        prompt_key,
        tokenizer,
    ):
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self._gangs = gangs
        self._context = context
        self.reward_model = reward_model
        self.reward_name = reward_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def format_prompt(self, prompt_text, rollout_text):
        messages = [
            {
                "role": "user",
                "content": prompt_text,
            },
            {
                "role": "assistant",
                "content": rollout_text,
            },
        ]

        return messages

    @override
    def process_rollouts(
        self, vllm_outputs: list[RequestOutput], prompt_batch: PromptBatch
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
                vllm_input = self.format_prompt(prompt_text, rollout_text)
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


class GenerativePointwiseVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        if reward_config.tokenizer is None:
            raise RuntimeError("Generative judges require tokenizer")

        if reward_config.judgment_extractor is None:
            raise RuntimeError(
                "Generative judges require implementing and specifying a judgment extractor"
            )

        return GenerativePointwiseVerifier(
            gangs,
            context,
            reward_model,
            reward_name,
            judgment_extractor=reward_config.judgment_extractor,
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            tokenizer=reward_config.tokenizer,
        )

    @property
    @override
    def name(self):
        return "generative_pointwise_verifier"

    @property
    @override
    def config_kls(self):
        return None


class GenerativePointwiseVerifier(VLLMOutputReward):
    def __init__(
        self,
        gangs,
        context,
        reward_model,
        reward_name,
        judgment_extractor,
        answer_key,
        prompt_key,
        tokenizer,
    ):
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self._gangs = gangs
        self._context = context
        self.reward_model = reward_model
        self.reward_name = reward_name
        self.judgment_extractor = judgment_extractor
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        judgment_extractor_registry = self._context.get_registry(
            JudgmentExtractorHandler
        )
        judgment_extractor_handler = judgment_extractor_registry.get(judgment_extractor)
        self.judgment_extractor = judgment_extractor_handler.create()

    @override
    def process_rollouts(
        self, vllm_outputs: list[RequestOutput], prompt_batch: PromptBatch
    ):
        vllm_inputs = []
        batch_text = []
        batch_tokens = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        text_prompts = prompt_batch.meta_info.get(self.prompt_key)
        reference_answers = prompt_batch.meta_info.get(self.answer_key)
        for i, (i_batch_request_output, prompt_text) in enumerate(
            zip(vllm_outputs, text_prompts)
        ):
            rollouts_text = []
            rollouts_tokens = []
            i_reference_answer = reference_answers[i]
            for rollout_output in i_batch_request_output.outputs:
                rollout_text = rollout_output.text
                vllm_input = self.judgment_extractor.format_prompt(
                    prompt_text, rollout_text, i_reference_answer
                )
                vllm_inputs.append(vllm_input)
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)

            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)

        batch_judgments = generate_rewards_generative(
            vllm_inputs, dp_gang=self._gangs.dp, vllm_model=self.reward_model
        )

        batch_rewards = []
        for per_rollout_judgments in batch_judgments:
            per_rollout_rewards = [
                self.judgment_extractor.extract(judgment.text)
                for judgment in per_rollout_judgments.outputs
            ]
            batch_rewards.append(self.judgment_extractor.aggregate(per_rollout_rewards))

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


class GenerativePairwiseVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        if reward_config.tokenizer is None:
            raise RuntimeError("Generative judges require tokenizer")

        if reward_config.judgment_extractor is None:
            raise RuntimeError(
                "Generative judges require implementing and specifying a judgment extractor"
            )

        return GenerativePairwiseVerifier(
            gangs,
            context,
            reward_model,
            reward_name,
            judgment_extractor=reward_config.judgment_extractor,
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            tokenizer=reward_config.tokenizer,
        )

    @property
    @override
    def name(self):
        return "generative_pairwise_verifier"

    @property
    @override
    def config_kls(self):
        return None


class GenerativePairwiseVerifier(VLLMOutputReward):
    def __init__(
        self,
        gangs,
        context,
        reward_model,
        reward_name,
        judgment_extractor,
        answer_key,
        prompt_key,
        tokenizer,
    ):
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self._gangs = gangs
        self._context = context
        self.reward_model = reward_model
        self.reward_name = reward_name
        self.judgment_extractor = judgment_extractor
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        judgment_extractor_registry = self._context.get_registry(
            JudgmentExtractorHandler
        )
        judgment_extractor_handler = judgment_extractor_registry.get(judgment_extractor)
        self.judgment_extractor = judgment_extractor_handler.create()

    @override
    def process_rollouts(
        self, vllm_outputs: list[RequestOutput], prompt_batch: PromptBatch
    ):
        vllm_inputs = []
        batch_text = []
        batch_tokens = []
        batch_pairwise_indices = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        text_prompts = prompt_batch.meta_info.get(self.prompt_key)
        for i, (i_batch_request_output, prompt_text) in enumerate(
            zip(vllm_outputs, text_prompts)
        ):
            rollouts_text = [
                rollout_output.text for rollout_output in i_batch_request_output.outputs
            ]
            rollouts_tokens = [
                rollout_output.token_ids
                for rollout_output in i_batch_request_output.outputs
            ]
            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)

            prompt_pairwise_indices = []
            for a in range(len(i_batch_request_output.outputs)):
                for b in range(len(i_batch_request_output.outputs)):
                    if a != b:
                        rollout_A_text = i_batch_request_output.outputs[a].text
                        rollout_B_text = i_batch_request_output.outputs[b].text
                        vllm_input = self.judgment_extractor.format_prompt(
                            prompt_text, rollout_A_text, rollout_B_text
                        )
                        vllm_inputs.append(vllm_input)
                        prompt_pairwise_indices.append((a, b))

            batch_pairwise_indices.append(prompt_pairwise_indices)

        batch_pairwise_judgments = generate_rewards_generative(
            vllm_inputs,
            dp_gang=self._gangs.dp,
            vllm_model=self.reward_model,
        )

        batch_pairwise_rewards = []
        for per_rollout_judgments in batch_pairwise_judgments:
            per_rollout_rewards = [
                self.judgment_extractor.extract(judgment.text)
                for judgment in per_rollout_judgments.outputs
            ]
            batch_pairwise_rewards.append(
                self.judgment_extractor.aggregate(per_rollout_rewards)
            )

        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts

        # Logic to convert pairwise scores into pointwise rewards
        # Can be done differently too
        batch_rewards = []
        for i in range(B):
            prompt_pairwise_rewards = batch_pairwise_rewards[
                i * R * (R - 1) : (i + 1) * R * (R - 1)
            ]
            prompt_pairwise_indices = batch_pairwise_indices[i]
            prompt_rewards = [0.0] * R
            for index, rewards in zip(prompt_pairwise_indices, prompt_pairwise_rewards):
                prompt_rewards[index[0]] += rewards[0]
                prompt_rewards[index[1]] += rewards[1]

            # Average score over 2*(R-1) pairwise comparisons
            if (R - 1) > 0:
                prompt_rewards = [
                    round(prompt_reward / (2 * (R - 1)), 4)
                    for prompt_reward in prompt_rewards
                ]

            batch_rewards.append(prompt_rewards)

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


class PplDerivedVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(
        self,
        reward_model: Any,
        reward_name: str,
        reward_config: object,
        gangs: Gangs,
        context,
    ) -> VLLMOutputReward:
        assert (
            reward_config.tokenizer is not None
        ), "Ppl Drived Verifier requires a tokenizer"

        return PplDerivedVerifier(
            gangs,
            context,
            reward_model,
            reward_name,
            # judgment_extractor=reward_config.judgment_extractor,
            answer_key=reward_config.answer_key,
            tokenizer=reward_config.tokenizer,
            apply_ppl_diff_reward=reward_config.additional_fields.get(
                "apply_ppl_diff_reward", True
            ),
            completion_window=reward_config.additional_fields.get(
                "completion_window", 100
            ),
            reward_formula=reward_config.additional_fields.get(
                "reward_formula", "diff"
            ),
            n_reason_mask=reward_config.additional_fields.get("n_reason_mask", 0),
            wrap_think_tags=reward_config.additional_fields.get(
                "wrap_think_tags", True
            ),
        )

    @property
    @override
    def name(self) -> str:
        return "ppl_derived_verifier"

    @property
    @override
    def config_kls(self):
        return None


class PplDerivedVerifier(VLLMOutputReward):
    def __init__(
        self,
        gangs,
        context,
        reward_model,
        reward_name,
        answer_key,
        tokenizer,
        apply_ppl_diff_reward,
        completion_window,
        reward_formula,
        n_reason_mask,
        wrap_think_tags,
        reason_start_wrap_key="reason_start_wrap",
        reason_end_wrap_key="reason_end_wrap",
    ):
        self.answer_key = answer_key
        self._gangs = gangs
        self._context = context
        self.reward_model = reward_model
        self.reward_name = reward_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.apply_ppl_diff_reward = apply_ppl_diff_reward
        self.completion_window = completion_window
        self.reward_formula = reward_formula
        self.n_reason_mask = n_reason_mask
        self.enable_human_friendly_log = True
        self.reason_start_wrap_key = reason_start_wrap_key
        self.reason_end_wrap_key = reason_end_wrap_key
        self.wrap_think_tags = wrap_think_tags

    def _preprocess_reward_input(
        self,
        prefix_tokens: List[int],
        reason: Optional[str],
        completion: str,
        n_prefix_truncate: Optional[int] = None,
        completion_window: int = 100,
        # if this is not None, we mask n_reason_mask tokens where we generate
        # the rollouts and compute reward on future tokens after them. i.e.
        # changing from next token reasoning -> future token reasoning
        completion_fmt: str = "{completion}",
        reason_fmt: str = "{reason}{reason_end_wrap}",
        reason_start_wrap: str = "",
        reason_end_wrap: str = "",
    ):
        # no reasoning augmented.
        if reason is None or not self.wrap_think_tags:
            prefix_text = self.tokenizer.decode(prefix_tokens, add_special_token=False)
            prefix_text = (
                prefix_text.removesuffix(" <think>").removesuffix("<think>")
                if reason_start_wrap == ""
                else prefix_text.removesuffix(reason_start_wrap)
            )
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        n_prefix_tokens = len(prefix_tokens)
        n_prefix_truncate = (
            n_prefix_tokens
            if n_prefix_truncate is None
            else min(n_prefix_truncate, n_prefix_tokens)
        )
        completion_tokens = self.tokenizer.encode(
            completion_fmt.format(completion=completion), add_special_tokens=False
        )
        if reason is None:
            text_tokens = prefix_tokens + completion_tokens
            n_input_tokens = n_prefix_truncate + self.n_reason_mask
            # prefix + groundtruth at masked positions -> future tokens
            n_input_tokens_all = n_prefix_tokens + self.n_reason_mask
        else:
            # if not reason.endswith("</think>"):
            #     reason += "</think>"  # add the stop token itself. so we have the pair of <think> </think>
            reason_tokens = self.tokenizer.encode(
                reason_fmt.format(
                    reason=reason,
                    reason_end_wrap=reason_end_wrap if self.wrap_think_tags else "",
                ),
                add_special_tokens=False,
            )
            text_tokens = (
                prefix_tokens + reason_tokens + completion_tokens[self.n_reason_mask :]
            )
            n_input_tokens = n_prefix_truncate + len(reason_tokens)
            n_input_tokens_all = n_prefix_tokens + len(reason_tokens)

        text_tokens = text_tokens[
            n_input_tokens_all - n_input_tokens : n_input_tokens_all + completion_window
        ]
        return text_tokens, n_input_tokens

    def extract_reward(self, prompt_logprobs: List[Any], prompt_len: int) -> float:
        completion_logprobs = prompt_logprobs[prompt_len:]
        completion_logprobs_vals = [
            list(d.values())[0].logprob for d in completion_logprobs
        ]  # TODO: douable check we skip the first None
        mean_logp = sum(completion_logprobs_vals) / len(completion_logprobs_vals)
        ppl = math.exp(-mean_logp)
        return -ppl  # negative ppl as reward

    def aggregate(self, rewards):
        if self.apply_ppl_diff_reward:
            if self.reward_formula == "diff":
                return [
                    reward - rewards[0] for reward in rewards[1:]
                ]  # NOTE: order is without reasoning augmentation, with reasoning augmentation -ppl
            elif self.reward_formula == "diff_percent":
                return [
                    (reward - rewards[0]) / (-rewards[0]) * 100
                    for reward in rewards[1:]
                ]
            else:
                raise NotImplementedError
        else:
            return rewards  # reason augmented case: -ppl as reward

    def _log_human_friendly(
        self,
        B,
        tokenizer,
        rm_vllm_inputs,  # flatten
        all_input_tok_lens,  # flatten
        rewards,  # flatten
        rm_rollouts,  # flatten
    ):
        assert (
            len(rm_vllm_inputs) == len(all_input_tok_lens)
            and len(all_input_tok_lens) == len(rewards)
            and len(rewards) == len(rm_rollouts)
        )
        N = len(rm_vllm_inputs)
        ex_n = N // B
        # group into a 2d list with the first dim equals B
        rm_vllm_inputs_batch = [
            rm_vllm_inputs[i * ex_n : (i + 1) * ex_n] for i in range(B)
        ]
        input_tok_lens_batch = [
            all_input_tok_lens[i * ex_n : (i + 1) * ex_n] for i in range(B)
        ]
        rewards_batch = [rewards[i * ex_n : (i + 1) * ex_n] for i in range(B)]
        rm_rollouts_batch = [rm_rollouts[i * ex_n : (i + 1) * ex_n] for i in range(B)]

        for example_i, (
            ex_rm_vllm_inputs,
            ex_input_tok_lens,
            ex_rewards,
            ex_rm_rollouts,
        ) in enumerate(
            zip(
                rm_vllm_inputs_batch,
                input_tok_lens_batch,
                rewards_batch,
                rm_rollouts_batch,
            )
        ):
            log.info("=" * 6 + f"example {example_i} summary" + "=" * 6)
            for i, (tokens, prefix_len, rm_rollout, reward) in enumerate(
                zip(ex_rm_vllm_inputs, ex_input_tok_lens, ex_rm_rollouts, ex_rewards)
            ):  # individual rollouts
                log.info("-" * 6 + f"prefix {i}" + "-" * 6)
                log.info(
                    tokenizer.decode(tokens[:prefix_len], skip_special_tokens=True)
                )
                if self.n_reason_mask > 0 and i == 0:
                    log.info("-" * 6 + f"masked tokens groundtruth {i}" + "-" * 6)
                    log.info(
                        tokenizer.decode(
                            tokens[prefix_len - self.n_reason_mask : prefix_len],
                            skip_special_tokens=True,
                        )
                    )
                log.info("-" * 6 + f"completion window {i}" + "-" * 6)
                log.info(
                    tokenizer.decode(tokens[prefix_len:], skip_special_tokens=True)
                )
                completion_logprobs = rm_rollout.prompt_logprobs[prefix_len:]
                completion_logprobs_vals: List[float] = [
                    list(d.values())[0].logprob for d in completion_logprobs
                ]
                log.info("-" * 6 + f"completion logprobs {i}" + "-" * 6)
                log.info(str(completion_logprobs_vals))
                completion_logprobs = [list(d.values())[0] for d in completion_logprobs]
                log.info("-" * 6 + f"completion token logprobs {i}" + "-" * 6)
                log.info(str(completion_logprobs))
                log.info("-" * 6 + f"reward {i}" + "-" * 6)
                log.info(reward)
            log.info("-" * 6 + "all rewards in example" + "-" * 6)
            log.info(ex_rewards)

    @override
    def process_rollouts(
        self,
        vllm_outputs: list[RequestOutput],
        prompt_batch: PromptBatch,
    ):
        all_input_tok_lens = []
        vllm_inputs = []
        batch_text = []
        batch_tokens = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        completion_batch = prompt_batch.meta_info.get(self.answer_key)
        reason_start_wrap_batch = prompt_batch.meta_info.get(self.reason_start_wrap_key)
        reason_end_wrap_batch = prompt_batch.meta_info.get(self.reason_end_wrap_key)

        log.debug(f"{vllm_outputs=}")
        log.debug(f"{completion_batch=}")

        for prefix, vllm_output, completion, reason_start_wrap, reason_end_wrap in zip(
            prompt_batch.prompts,
            vllm_outputs,
            completion_batch,
            reason_start_wrap_batch,
            reason_end_wrap_batch,
        ):
            rollouts_text = []
            rollouts_tokens = []

            # case where no reasoning is augmented
            if self.apply_ppl_diff_reward:
                text_tokens, n_input_tokens = self._preprocess_reward_input(
                    prefix,
                    None,
                    completion,
                    completion_window=self.completion_window,
                    reason_start_wrap=reason_start_wrap,
                    reason_end_wrap=reason_end_wrap,
                )
                # log.debug(f"{text_tokens=}, {n_input_tokens=}")
                all_input_tok_lens.append(n_input_tokens)
                vllm_inputs.append(text_tokens)

            for rollout_output in vllm_output.outputs:  # reasoning in rollouts
                text_tokens, n_input_tokens = self._preprocess_reward_input(
                    prefix,
                    rollout_output.text,
                    completion,
                    completion_window=self.completion_window,
                    reason_start_wrap=reason_start_wrap,
                    reason_end_wrap=reason_end_wrap,
                )
                # log.debug(f"{text_tokens=}, {n_input_tokens=}")
                all_input_tok_lens.append(n_input_tokens)
                vllm_inputs.append(text_tokens)
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)

            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)
        log.debug(f"{vllm_inputs=}")
        log.debug(f"{all_input_tok_lens=}")

        rm_sampling_params = {
            "n": 1,
            "max_tokens": 1,
            "prompt_logprobs": 0,
            "detokenize": self.enable_human_friendly_log,
        }
        rollouts = generate_rollouts(
            vllm_inputs,
            dp_gang=self._gangs.dp,
            vllm_model=self.reward_model,
            sampling_params=SamplingParams(**rm_sampling_params),
            string_input=False,
        )

        curr_rewards = [
            self.extract_reward(rollout.prompt_logprobs, prompt_len=input_len)
            for rollout, input_len in zip(rollouts, all_input_tok_lens)
        ]
        log.info(f"{curr_rewards=}")

        batch_rewards = self.aggregate(curr_rewards)
        log.info(f"{batch_rewards=}")

        # reshape batch_rewards to [Batch, Rollouts]
        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts
        batch_rewards = [batch_rewards[i * R : (i + 1) * R] for i in range(B)]
        if self.enable_human_friendly_log:
            self._log_human_friendly(
                B,
                self.tokenizer,
                vllm_inputs,
                all_input_tok_lens,
                curr_rewards,
                rollouts,
            )

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


class PrefixMatchVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(
        self,
        reward_model: Any,
        reward_name: str,
        reward_config: object,
        gangs: Gangs,
        context,
    ) -> VLLMOutputReward:
        assert (
            reward_config.tokenizer is not None
        ), "Ppl Drived Verifier requires a tokenizer"

        return PrefixMatchVerifier(
            gangs,
            context,
            reward_name,
            answer_key=reward_config.answer_key,
            tokenizer=reward_config.tokenizer,
        )

    @property
    @override
    def name(self) -> str:
        return "prefix_match_verifier"

    @property
    @override
    def config_kls(self):
        return None


class PrefixMatchVerifier(VLLMOutputReward):
    def __init__(
        self,
        gangs,
        context,
        reward_name,
        answer_key,
        tokenizer,
    ):
        self.answer_key = answer_key
        self._gangs = gangs
        self._context = context
        self.reward_name = reward_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def prefix_match_reward(self, predicted: str, groundtruth: str) -> float:
        gt_tokens = self.tokenizer.encode(groundtruth, add_special_tokens=False)
        gt_prefix = ""
        for token in gt_tokens:
            gt_prefix += self.tokenizer.decode(token, add_special_tokens=False)
            if predicted == gt_prefix:
                return 1
            elif len(predicted) < len(gt_prefix):
                return 0

        return 0

    def extract_reward(self, response: str, groundtruth: str) -> float:
        MAX_GT_BYTES = 300
        groundtruth = groundtruth[:MAX_GT_BYTES]
        match = re.search(r"\\boxed\{(.*?)\}", response)
        predicted = match.group(1) if match else None
        log.debug(f"{groundtruth=}")
        log.debug(f"{response=}")
        log.debug(f"{predicted=}")
        if predicted is None:
            return 0
        else:
            return self.prefix_match_reward(predicted, groundtruth)

    @override
    def process_rollouts(
        self,
        vllm_outputs: list[RequestOutput],
        prompt_batch: PromptBatch,
    ):
        batch_text = []
        batch_tokens = []
        batch_rewards = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        completion_batch = prompt_batch.meta_info.get(self.answer_key)

        # log.debug(f"{vllm_outputs=}")
        # log.debug(f"{completion_batch=}")
        if log.is_enabled_for_debug():
            prompt_text = [
                self.tokenizer.decode(prompt, add_special_token=False)
                for prompt in prompt_batch.prompts
            ]
            log.debug(f"{prompt_text=}")

        for vllm_output, completion in zip(vllm_outputs, completion_batch):
            rollouts_text = []
            rollouts_tokens = []
            rewards = []
            log.debug(f"{vllm_output.prompt_token_ids=}")

            for rollout_output in vllm_output.outputs:  # reasoning in rollouts
                rewards.append(self.extract_reward(rollout_output.text, completion))
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)

            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)
            batch_rewards.append(rewards)
        log.debug(f"{batch_rewards=}")

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:
        pass
