# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from fairseq2.logging import log
import random

import torch
from transformers import AutoTokenizer
from typing_extensions import override
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams

from fairseq2.context import RuntimeContext
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gangs
from fairseq2.recipes.lm._online_finetune._common import (
    _mute_output,
    collate_with_target_mask,
    generate_rewards,
    generate_rewards_generative,
    prepare_preference_batch_random_pair,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    JudgmentExtractorHandler,
)


@dataclass(kw_only=True)
class RewardModelConfig:
    answer_key: str = "answer"
    prompt_key: str = "prompt"
    tokenizer: str | None = None
    judgment_extractor: str | None = None


@dataclass(kw_only=True)
class RewardSection:
    name: str = "dummy"
    config: RewardModelConfig = field(default_factory=lambda: RewardModelConfig())


class VLLMOutputRewardHandler(ABC):
    @abstractmethod
    def create(
        self, reward_model: Any, gangs: Gangs, reward_config: object
    ) -> VLLMOutputReward:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]:
        ...


class VLLMOutputReward(ABC):
    @abstractmethod
    def process_rollouts(self, vllm_outputs: list[RequestOutput]):
        ...

    @abstractmethod
    def prepare_preference_batch(self, prompt_batch: PromptBatch, rollouts):
        ...


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

        batch, is_bad_batch = prepare_preference_batch_random_pair(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

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

        batch, is_bad_batch = prepare_preference_batch_random_pair(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        return batch, is_bad_batch, reward_output


class SkyworkVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        if reward_config.tokenizer is not None:
            tokenizer = reward_config.tokenizer
        else:
            tokenizer = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

        return SkyworkVerifier(
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
        return "skywork_verifier"

    @property
    @override
    def config_kls(self):
        return None


class SkyworkVerifier(VLLMOutputReward):
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

    def wrap_text(self, prompt_text, rollout_text):
        wrapped_text = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": rollout_text},
        ]
        chat_str = self.tokenizer.apply_chat_template(wrapped_text, tokenize=False)
        if self.tokenizer.bos_token is not None and chat_str.startswith(
            self.tokenizer.bos_token
        ):
            chat_str = chat_str[len(self.tokenizer.bos_token) :]

        return chat_str

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
        self.judgment_extractor = judgment_extractor_handler.create(self.tokenizer)

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
            for rollout_output in i_batch_request_output.outputs:
                rollout_text = rollout_output.text
                if reference_answers is None:
                    vllm_input = self.judgment_extractor.format_prompt(
                        prompt_text, rollout_text
                    )
                else:
                    vllm_input = self.judgment_extractor.format_prompt(
                        prompt_text, rollout_text, reference_answers[i]
                    )
                vllm_inputs.append(vllm_input)
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)

            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)

        batch_judgments = generate_rewards_generative(
            vllm_inputs, dp_gang=self._gangs.dp, vllm_model=self.reward_model
        )
        
        log.info(f"Sample judgment: {batch_judgments[0].outputs[0].text}")

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
        self.judgment_extractor = judgment_extractor_handler.create(self.tokenizer)
        
    def all_pairs(self, prompt_text, i_batch_request_output, vllm_inputs, prompt_pairwise_indices, reference_answer=None):
        for a in range(len(i_batch_request_output.outputs)):
            for b in range(len(i_batch_request_output.outputs)):
                if a != b:
                    rollout_A_text = i_batch_request_output.outputs[a].text
                    rollout_B_text = i_batch_request_output.outputs[b].text
                    vllm_input = self.judgment_extractor.format_prompt(
                        prompt_text, rollout_A_text, rollout_B_text, reference_answer
                    )
                    vllm_inputs.append(vllm_input)
                    prompt_pairwise_indices.append((a, b))
                    
        return vllm_inputs, prompt_pairwise_indices
    
    def pairs_with_reference(self, prompt_text, i_batch_request_output, vllm_inputs, prompt_pairwise_indices, reference_answer=None):
        reference_idx = random.randint(0, len(i_batch_request_output.outputs)-1)
        reference_rollout = i_batch_request_output.outputs[reference_idx].text
        for a in range(len(i_batch_request_output.outputs)):
            if a != reference_idx:
                rollout_A_text = i_batch_request_output.outputs[a].text
                rollout_B_text = reference_rollout
                
                to_swap = random.choice([True, False])
                if to_swap:
                    rollout_A_text, rollout_B_text = rollout_B_text, rollout_A_text
                    prompt_pairwise_indices.append((reference_idx, a))
                else:
                    prompt_pairwise_indices.append((a, reference_idx))          
                vllm_input = self.judgment_extractor.format_prompt(
                    prompt_text, rollout_A_text, rollout_B_text, reference_answer
                )
                vllm_inputs.append(vllm_input)
                
        return vllm_inputs, prompt_pairwise_indices
    
    def random_pairs(self, prompt_text, i_batch_request_output, vllm_inputs, prompt_pairwise_indices, reference_answer=None):
        all_pairs = [(i, j) for i in range(len(i_batch_request_output.outputs)) for j in range(len(i_batch_request_output.outputs)) if i != j]
        random_pairs = random.sample(all_pairs, len(i_batch_request_output.outputs))
        
        for a in range(len(i_batch_request_output.outputs)):
            for b in range(len(i_batch_request_output.outputs)):
                if (a, b) in random_pairs:
                    rollout_A_text = i_batch_request_output.outputs[a].text
                    rollout_B_text = i_batch_request_output.outputs[b].text
                    vllm_input = self.judgment_extractor.format_prompt(
                        prompt_text, rollout_A_text, rollout_B_text, reference_answer
                    )
                    vllm_inputs.append(vllm_input)
                    prompt_pairwise_indices.append((a, b))
                    
        return vllm_inputs, prompt_pairwise_indices
    
    def convert_pairwise_rewards_to_pointwise(self, batch_pairwise_rewards, batch_pairwise_indices, batch_text, batch_type):
        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts
        batch_pointwise_rewards = []
        
        for i in range(B):
            # Extract the pairwise rewards for each input
            if batch_type == "all_pairs":
                prompt_pairwise_rewards = batch_pairwise_rewards[
                    i * R * (R - 1) : (i + 1) * R * (R - 1)
                ]
            elif batch_type == "reference":
                prompt_pairwise_rewards = batch_pairwise_rewards[
                    i * (R - 1) : (i + 1) * (R - 1)
                ]
            elif batch_type == "random_pairs":
                prompt_pairwise_rewards = batch_pairwise_rewards[
                    i * R : (i + 1) * R
                ]
                
            # Sum the rewards for each rollout and count how many times each rollout appears in pairwise judgments
            prompt_pairwise_indices = batch_pairwise_indices[i]
            prompt_rewards = [0.0] * R
            counts = [0] * R
            for index, rewards in zip(prompt_pairwise_indices, prompt_pairwise_rewards):
                prompt_rewards[index[0]] += rewards[0]
                prompt_rewards[index[1]] += rewards[1]
                counts[index[0]] += 1
                counts[index[1]] += 1

            # Compute average pointwise rewards 
            avg_prompt_rewards = [0.0] * R
            for i in range(len(prompt_rewards)):
                if counts[i] > 0:
                    avg_prompt_rewards[i] = round(prompt_rewards[i] / counts[i], 4)

            batch_pointwise_rewards.append(avg_prompt_rewards)
            
        return batch_pointwise_rewards

    @override
    def process_rollouts(
        self, vllm_outputs: list[RequestOutput], prompt_batch: PromptBatch
    ):
        vllm_inputs = []
        batch_text = []
        batch_tokens = []
        batch_pairwise_indices = []
        
        batch_type = "random_pairs"

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        text_prompts = prompt_batch.meta_info.get(self.prompt_key)
        try:
            reference_answers = prompt_batch.meta_info.get(self.answer_key)
        except:
            reference_answers = [None] * len(prompt_batch.prompts)
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
            if batch_type == "all_pairs":
                vllm_inputs, prompt_pairwise_indices = self.all_pairs(prompt_text, i_batch_request_output, vllm_inputs, prompt_pairwise_indices, reference_answers[i])
            elif batch_type == "reference":
                vllm_inputs, prompt_pairwise_indices = self.pairs_with_reference(prompt_text, i_batch_request_output, vllm_inputs, prompt_pairwise_indices, reference_answers[i])
            elif batch_type == "random_pairs":
                vllm_inputs, prompt_pairwise_indices = self.random_pairs(prompt_text, i_batch_request_output, vllm_inputs, prompt_pairwise_indices, reference_answers[i])

            batch_pairwise_indices.append(prompt_pairwise_indices)

        batch_pairwise_judgments = generate_rewards_generative(
            vllm_inputs,
            dp_gang=self._gangs.dp,
            vllm_model=self.reward_model,
        )
        
        log.info(f"Here: {len(batch_pairwise_judgments)}")
        log.info(f"Here: {len(batch_pairwise_judgments[0].outputs)}")
        log.info(f"Sample judgment: {batch_pairwise_judgments[0].outputs[0].text}")

        batch_pairwise_rewards = []
        for per_rollout_judgments in batch_pairwise_judgments:
            per_rollout_rewards = [
                self.judgment_extractor.extract(judgment.text)
                for judgment in per_rollout_judgments.outputs
            ]
            batch_pairwise_rewards.append(
                self.judgment_extractor.aggregate(per_rollout_rewards)
            )

        batch_rewards = self.convert_pairwise_rewards_to_pointwise(batch_pairwise_rewards, batch_pairwise_indices, batch_text, batch_type)
        
        log.info(f"Batch Rewards: {batch_rewards}")
        
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
