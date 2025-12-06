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
from typing import Any, Dict, List, Optional, Tuple

import torch
from fairseq2.context import RuntimeContext
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gangs
from fairseq2.logging import log as fs2_log
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

log = logging.getLogger(__name__)
# fs2_log._logger.setLevel(logging.DEBUG)


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

        self.verify_func = math_metric(
            gold_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(boxed_match_priority=0),
            ),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(boxed_match_priority=0),
            ),
            aggregation_function=max,
            precision=6,
        )

    def remove_think_string(self, completion: str):
        # Remove reasoning from completion before verification
        # If both <think> and </think> exist, remove everything between them
        # If only </think> exists, remove everything before and including it
        if "<think>" in completion and "</think>" in completion:
            start_idx = completion.find("<think>")
            end_idx = completion.find("</think>") + len("</think>")
            completion = completion[:start_idx] + completion[end_idx:]
        elif "</think>" in completion:
            end_idx = completion.find("</think>") + len("</think>")
            completion = completion[end_idx:]
        elif "<think>" in completion:
            completion = ""
        return completion

    def verify_answer(self, completion: str, answer: str):
        completion = self.remove_think_string(completion)

        # here we add extra $$ to label so that LatexExtractor works as expected
        # if answer doesn't contain \\boxed, we add it
        # if not answer.startswith("$"):
        if "\\boxed" not in answer:
            # answer = f"${answer}$"
            answer = "\\boxed{" + answer + "}"
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

        text_prompts = prompt_batch.meta_info.get("raw_prompt_text")
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
                    self.tokenizer,
                    prompt_text,
                    rollout_text,
                    i_reference_answer,
                    self._gangs.dp,
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

        # Log prompt_text, i_reference_answer, rollout_text, and per_rollout_reward together
        rollout_idx = 0
        for i, (prompt_text, i_reference_answer) in enumerate(
            zip(text_prompts, reference_answers)
        ):
            for rollout_text in batch_text[i]:
                per_rollout_reward = batch_rewards[rollout_idx]

                # Split rollout_text into think and gen_suffix based on </think> token
                think_tag = "</think>"
                if think_tag in rollout_text:
                    think_end_idx = rollout_text.find(think_tag) + len(think_tag)
                    gen_think = rollout_text[:think_end_idx]
                    gen_suffix = rollout_text[think_end_idx:]
                else:
                    gen_think = ""
                    gen_suffix = rollout_text

                log.info("====================================================")
                log.info(f"Prefix = {prompt_text}")
                log.info(f"[Think Start]\n{gen_think}\n[Think End]")
                log.info(
                    f"\n[Gold Suffix Start]\n{i_reference_answer}\n[Gold Suffix End]\n\n[Gen Suffix Start]\n{gen_suffix}\n[Gen Suffix End]"
                )
                log.info(f"Score = {per_rollout_reward}")
                rollout_idx += 1

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
            prompt_key=reward_config.prompt_key,
            answer_key=reward_config.answer_key,
            tokenizer=reward_config.tokenizer,
            reward_type=reward_config.additional_fields.get("reward_type"),
            completion_window=reward_config.additional_fields.get(
                "completion_window", 100
            ),
            n_reason_mask=reward_config.additional_fields.get("n_reason_mask", 0),
            wrap_think_tags=reward_config.additional_fields.get(
                "wrap_think_tags", True
            ),
            vllm_proc_name_dict=reward_config.additional_fields.get(
                "vllm_proc_name_dict", None
            ),
            reward_agg=reward_config.additional_fields.get("reward_agg", "sum"),
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
        prompt_key,
        answer_key,
        tokenizer,
        reward_type,
        completion_window,
        n_reason_mask,
        wrap_think_tags,
        reason_start_wrap_key="reason_start_wrap",
        reason_end_wrap_key="reason_end_wrap",
        vllm_proc_name_dict=None,
        reward_agg="sum",
    ):
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self._gangs = gangs
        self._context = context
        self.reward_model = reward_model
        self.reward_name = reward_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.reward_type = reward_type
        self.completion_window = completion_window
        self.n_reason_mask = n_reason_mask
        self.reason_start_wrap_key = reason_start_wrap_key
        self.reason_end_wrap_key = reason_end_wrap_key
        self.wrap_think_tags = wrap_think_tags
        self.enable_human_friendly_log = False
        self.vllm_proc_name_dict = vllm_proc_name_dict
        self.reward_agg = reward_agg

        self._dummy_prefix_w_think_tag = "Water in a pan reaches 100°C, but the pan is still left on the heat, so eventually all of the water turns to water vapor. Calculate the energy needed to evaporate the 1.2 kg of water contained by the pan. Use a value of 2258 kJ/kg for the specific latent heat of vaporization of water. Give your answer to 2 significant figures. <think>"
        self._dummy_suffix = "04:11 ### Video Transcript Water in a pan reaches 100 degrees Celsius. But the pan is still left on the heat. So eventually, all of the water turns to water vapor. Calculate the energy needed to evaporate the 1.2 kilograms of water contained by the pan. Use a value of 2258 kilojoules per kilogram for the specific latent heat of vaporization of water. Give your answer to two significant figures. Alright. So this is a long question. So we should start by underlining all the important"

        self.vllm_input_proc_dict = {
            # -------- prefix related --------
            "pref_to_wrapped_reason_for_rm": self._pref_to_wrapped_reason_for_rm,
            "wrapped_reason_for_rm": self._wrapped_reason_for_rm,
            "pref_soth_to_reason_eoth_for_rm": self._pref_soth_to_reason_eoth_for_rm,
            "dummy_pref_soth_to_reason_eoth_for_rm": self._dummy_pref_soth_to_reason_eoth_for_rm,
            "dummy_pref_to_reason_for_rm": self._dummy_pref_to_reason_for_rm,  # base
            "pref_to_reason_for_rm": self._pref_to_reason_for_rm,  # target
            "soth_to_reason_eoth_for_rm": self._soth_to_reason_eoth_for_rm,
            # -------- suffix related --------
            "pref_to_suf_for_rm": self._pref_to_suf_for_rm,  # base
            "pref_reason_to_suf_for_rm": self._pref_reason_to_suf_for_rm,  # target
            "pref_wrapped_reason_to_suf_for_rm": self._pref_wrapped_reason_to_suf_for_rm,  # target
            "dummy_pref_wrapped_reason_to_dummy_suf_for_rm": self._dummy_pref_wrapped_reason_to_dummy_suf_for_rm,  # base
        }
        self.reward_agg_fn_dict = {
            "cap_pref_0p01_weight_0p02_1": self._cap_pref_0p01_weight_0p02_1,
            "cap_pref_0p01_weight_0p05_1": self._cap_pref_0p01_weight_0p05_1,
            "cap_pref_0p02_weight_0p02_1": self._cap_pref_0p02_weight_0p02_1,
            "cap_pref_0p05_weight_0p02_1": self._cap_pref_0p05_weight_0p02_1,
            "cap_pref_0p08_weight_0p1_1": self._cap_pref_0p08_weight_0p1_1,
            "cap_pref_0p15_weight_0p2_1": self._cap_pref_0p15_weight_0p2_1,
            "cap_pref_0p5_weight_0p2_1": self._cap_pref_0p5_weight_0p2_1,
            "cap_pref_1_weight_0p2_1": self._cap_pref_1_weight_0p2_1,
            "weight_0p05_1": self._weight_0p05_1,
            "sum": (lambda x, y: x + y),
        }

    def __post_init__(self):
        # validate the input flag correctness.
        assert self.reward_type in ["logp_ratio", "logp", "ppl_ratio", "ppl"]
        assert "suffix_target" in self.vllm_proc_name_dict
        for key, proc_name in self.vllm_proc_name_dict.items():
            assert key in {
                "prefix_base",
                "prefix_target",
                "suffix_base",
                "suffix_target",
            }
            assert proc_name in self.vllm_input_proc_dict
        assert self.reward_agg in self.reward_agg_fn_dict

    def _tokenize(self, input_str, add_special_tokens=False, max_length=None):
        return (
            self.tokenizer.encode(input_str, add_special_tokens=add_special_tokens)
            if max_length is None
            else self.tokenizer.encode(
                input_str,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=True,
            )
        )

    def _concat_w_maybe_ws(self, left_text, right_text):
        if (not left_text[-1].isspace()) and (not right_text[0].isspace()):
            return left_text + " " + right_text
        return left_text + right_text

    def _get_rm_vllm_inputs(
        self,
        input_text_list: List[str] | str,
        target_text_list: List[str] | str,
        maybe_add_whitespace=True,
        target_window: List[int] = None,
    ) -> Tuple[List[int], int]:
        # this function returns the all tokens (input & target tokens) and length of input tokens for rm vllm.

        if isinstance(input_text_list, str):
            input_text_list = [input_text_list]
        if isinstance(target_text_list, str):
            target_text_list = [target_text_list]

        # input tokens
        input_tokens_list = [
            # TODO(lidli): check for qwen model
            self._tokenize(input_text, add_special_tokens=True)
            for input_text in input_text_list
        ]
        # target tokens
        target_tokens_list = [
            self._tokenize(target_text, max_length=target_window)
            for target_text in target_text_list
        ]

        assert (
            len(input_tokens_list) == 1
            or len(target_tokens_list) == 1
            or len(input_tokens_list) == len(target_tokens_list)
        )
        len_expanded = max(len(input_tokens_list), len(target_tokens_list))
        # expand
        if len(input_tokens_list) == 1:
            input_tokens_list = input_tokens_list * len_expanded
        if len(target_tokens_list) == 1:
            target_tokens_list = target_tokens_list * len_expanded
        if len(input_text_list) == 1:
            input_text_list = input_text_list * len_expanded
        if len(target_text_list) == 1:
            target_text_list = target_text_list * len_expanded

        results = []
        for input_text, target_text, input_tokens, target_tokens in zip(
            input_text_list, target_text_list, input_tokens_list, target_tokens_list
        ):
            # maybe add space between input and target texts
            if (
                maybe_add_whitespace
                and (not input_text[-1].isspace())
                and (not target_text[0].isspace())
            ):
                input_tokens = input_tokens + self._tokenize(" ")
            results.append((input_tokens + target_tokens, len(input_tokens)))
        return results

    def _pref_wrapped_reason_to_suf_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        # text of "prefix <think> reason </think> suffix" for rm(suffix | prefix <think> reason </think>)
        return self._get_rm_vllm_inputs(
            [
                inputs["prefix_w_think_tag"] + reason + inputs["think_end_tag"]
                for reason in inputs["think_texts"]
            ],
            inputs["suffix"],
            target_window=self.completion_window,
        )

    def _pref_reason_to_suf_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        # text of "prefix reason suffix" for rm(suffix | prefix reason)
        prefix = self._get_raw_prefix(
            inputs["prefix_w_think_tag"], inputs["think_start_tag"]
        )
        return self._get_rm_vllm_inputs(
            [
                self._concat_w_maybe_ws(prefix, reason)
                for reason in inputs["think_texts"]
            ],
            inputs["suffix"],
            target_window=self.completion_window,
        )

    def _get_dummy_prefix_w_think_tag(self) -> str:
        return self._dummy_prefix_w_think_tag

    def _get_dummy_suffix(self) -> str:
        return self._dummy_suffix

    def _dummy_pref_wrapped_reason_to_dummy_suf_for_rm(self, inputs: Dict[str, str]):
        # text of "dummy_prefix <think> reason </think> dummy_suffix" for rm(dummy_suffix | dummy_prefix <think> reason </think>)

        dummy_prefix_w_think_tag: str = self._get_dummy_prefix_w_think_tag()
        dummy_suffix: str = self._get_dummy_suffix()
        log.debug(f"{dummy_prefix_w_think_tag=}")
        log.debug(f"{dummy_suffix=}")
        assert (
            inputs["prefix_w_think_tag"] != dummy_prefix_w_think_tag
            and inputs["suffix"] != dummy_suffix
        )

        return self._get_rm_vllm_inputs(
            [
                dummy_prefix_w_think_tag + reason + inputs["think_end_tag"]
                for reason in inputs["think_texts"]
            ],
            dummy_suffix,
            target_window=self.completion_window,
        )

    def _get_raw_prefix(self, prefix_w_think_tag, think_start_tag):
        prefix = prefix_w_think_tag.removesuffix(think_start_tag)
        assert len(prefix) > 0, "empty prefix!"
        # remove upto 1 whitespace (to better the original text native whitespace.)
        if prefix[-1].isspace():
            prefix = prefix[:-1]
        return prefix

    def _pref_to_suf_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        # text of "prefix suffix" for rm(suffix | prefix)

        prefix = self._get_raw_prefix(
            inputs["prefix_w_think_tag"], inputs["think_start_tag"]
        )

        return self._get_rm_vllm_inputs(
            prefix, inputs["suffix"], target_window=self.completion_window
        )  # this is not reasoning dependent so only compute once.

    def _pref_to_wrapped_reason_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        # text of "prefix <think> reason </think>" for rm(<think> reason </think> | prefix)
        prefix = inputs["prefix_w_think_tag"].removesuffix(inputs["think_start_tag"])
        return self._get_rm_vllm_inputs(
            prefix,
            [
                inputs["think_start_tag"] + reason + inputs["think_end_tag"]
                for reason in inputs["think_texts"]
            ],
            maybe_add_whitespace=False,
        )

    def _wrapped_reason_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        return self._get_rm_vllm_inputs(
            "",
            [
                inputs["think_start_tag"] + reason + inputs["think_end_tag"]
                for reason in inputs["think_texts"]
            ],
            maybe_add_whitespace=False,
        )

    def _pref_soth_to_reason_eoth_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        # text: "prefix <think> reason </think>", for rm(reason </think> | prefix <think>)
        return self._get_rm_vllm_inputs(
            inputs["prefix_w_think_tag"],
            [reason + inputs["think_end_tag"] for reason in inputs["think_texts"]],
            maybe_add_whitespace=False,
        )

    def _dummy_pref_soth_to_reason_eoth_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        # text: "dummy_prefix <think> reason </think>", for rm(reason </think> | dummy_prefix <think>)
        dummy_prefix_w_think_tag: str = self._get_dummy_prefix_w_think_tag()
        log.debug(f"{dummy_prefix_w_think_tag=}")
        assert inputs["prefix_w_think_tag"] != dummy_prefix_w_think_tag
        return self._get_rm_vllm_inputs(
            dummy_prefix_w_think_tag,
            [reason + inputs["think_end_tag"] for reason in inputs["think_texts"]],
            maybe_add_whitespace=False,
        )

    def _dummy_pref_to_reason_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        # text: "dummy_prefix reason", for rm(reason | dummy_prefix)
        dummy_prefix_w_think_tag: str = self._get_dummy_prefix_w_think_tag()
        log.debug(f"{dummy_prefix_w_think_tag=}")
        assert inputs["prefix_w_think_tag"] != dummy_prefix_w_think_tag
        dummy_prefix = self._get_raw_prefix(
            dummy_prefix_w_think_tag, inputs["think_start_tag"]
        )
        return self._get_rm_vllm_inputs(dummy_prefix, inputs["think_texts"])

    def _pref_to_reason_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        # text: "dummy_prefix reason", for rm(reason | dummy_prefix)
        prefix = self._get_raw_prefix(
            inputs["prefix_w_think_tag"], inputs["think_start_tag"]
        )
        return self._get_rm_vllm_inputs(prefix, inputs["think_texts"])

    def _soth_to_reason_eoth_for_rm(
        self, inputs: Dict[str, str]
    ) -> List[Tuple[List[int], int]]:
        # text: "<think> reason </think>", for rm(reason </think> | <think>)
        return self._get_rm_vllm_inputs(
            inputs["think_start_tag"],
            [reason + inputs["think_end_tag"] for reason in inputs["think_texts"]],
            maybe_add_whitespace=False,
        )

    def _cap_pref_0p01_weight_0p05_1(self, pref_r, suffix_r):
        return min(0.01, pref_r) * 0.05 + suffix_r

    def _cap_pref_0p01_weight_0p02_1(self, pref_r, suffix_r):
        return min(0.01, pref_r) * 0.02 + suffix_r

    def _cap_pref_0p02_weight_0p02_1(self, pref_r, suffix_r):
        return min(0.02, pref_r) * 0.02 + suffix_r

    def _cap_pref_0p05_weight_0p02_1(self, pref_r, suffix_r):
        return min(0.05, pref_r) * 0.02 + suffix_r

    def _cap_pref_0p08_weight_0p1_1(self, pref_r, suffix_r):
        return min(0.08, pref_r) * 0.1 + suffix_r

    def _cap_pref_0p15_weight_0p2_1(self, pref_r, suffix_r):
        return min(0.15, pref_r) * 0.2 + suffix_r

    def _cap_pref_0p5_weight_0p2_1(self, pref_r, suffix_r):
        return min(0.5, pref_r) * 0.2 + suffix_r

    def _cap_pref_1_weight_0p2_1(self, pref_r, suffix_r):
        return min(1.0, pref_r) * 0.2 + suffix_r

    def _weight_0p05_1(self, pref_r, suffix_r):
        return pref_r * 0.05 + suffix_r

    # TODO(lidli): deprecate this and clear flags
    def _preprocess_reward_input(
        self,
        prefix: List[int] | str,
        reason: Optional[str],
        completion: str,
        n_prefix_truncate: Optional[int] = None,
        completion_window: int = 100,
        # if this is not None, we mask n_reason_mask tokens where we generate
        # the rollouts and compute reward on future tokens after them. i.e.
        # changing from next token reasoning -> future token reasoning
        completion_fmt: str = "{completion}",
        reason_fmt: str = "{reason}{reason_end_wrap}",
        reason_start_wrap: Optional[str] = None,
        reason_end_wrap: Optional[str] = None,
    ):
        # no reasoning augmented.
        prefix_text: str = (
            prefix
            if isinstance(prefix, str)
            else self.tokenizer.decode(prefix, add_special_token=False)
        )
        if reason is None or not self.wrap_think_tags:
            prefix_text = (
                prefix_text.removesuffix(" <think>").removesuffix("<think>")
                if reason_start_wrap is None
                else prefix_text.removesuffix(reason_start_wrap)
            )
        if reason is None:
            reason_text = None
        else:
            reason_end_wrap = reason_end_wrap or "</think>"
            reason_text: str = reason_fmt.format(
                reason=reason,
                reason_end_wrap=reason_end_wrap if self.wrap_think_tags else "",
            )
        completion_text: str = completion_fmt.format(completion=completion)

        # add whitespace if no whitespace between prefix and following text.
        text_after_prefix: str = (
            reason_text
            if ((reason_text is not None) and len(reason_text) > 0)
            else completion_text
        )
        if not (prefix_text[-1].isspace() or text_after_prefix[0].isspace()):
            prefix_text += " "

        # add whitespace to reason if needed
        if reason_text and (
            not (reason_text[-1].isspace() or completion_text[0].isspace())
        ):
            reason_text += " "

        prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        n_prefix_tokens: int = len(prefix_tokens)
        n_prefix_truncate = (
            n_prefix_tokens
            if n_prefix_truncate is None
            else min(n_prefix_truncate, n_prefix_tokens)
        )
        completion_tokens = self.tokenizer.encode(
            completion_text, add_special_tokens=False
        )
        if reason is None:
            text_tokens = prefix_tokens + completion_tokens
            n_input_tokens = n_prefix_truncate + self.n_reason_mask
            # prefix + groundtruth at masked positions -> future tokens
            n_input_tokens_all = n_prefix_tokens + self.n_reason_mask
        else:
            reason_tokens = self.tokenizer.encode(reason_text, add_special_tokens=False)
            text_tokens = (
                prefix_tokens + reason_tokens + completion_tokens[self.n_reason_mask :]
            )
            n_input_tokens = n_prefix_truncate + len(reason_tokens)
            n_input_tokens_all = n_prefix_tokens + len(reason_tokens)

        text_tokens = text_tokens[
            n_input_tokens_all - n_input_tokens : n_input_tokens_all + completion_window
        ]
        return text_tokens, n_input_tokens

    def extract_scores(self, prompt_logprobs: List[Any], prompt_len: int) -> float:
        completion_logprobs = prompt_logprobs[prompt_len:]
        assert completion_logprobs[0] is not None
        completion_logprobs_vals = [
            list(d.values())[0].logprob for d in completion_logprobs
        ]
        tokens = [list(item.keys())[0] for item in completion_logprobs]
        fs2_log.debug(f"completion tokens={tokens}")
        fs2_log.debug(f"{completion_logprobs_vals=}")
        mean_logp = sum(completion_logprobs_vals) / len(completion_logprobs_vals)
        if self.reward_type.startswith("logp"):
            return mean_logp
        elif self.reward_type.startswith("ppl"):
            # -ppl
            return -math.exp(-mean_logp)
        else:
            raise NotImplementedError

    def _cal_reward_diff(
        self, base_rewards: List[float], target_rewards: List[float], num_rollouts: int
    ) -> list[float]:
        if len(base_rewards) == 0:
            # rewards are not comparative in this case. And if both base and
            # target rewards are empty return zero rewards.
            return ([0] * num_rollouts) if target_rewards == [] else target_rewards
        else:
            # we avoid redundant computation earlier.
            if len(base_rewards) == 1:
                base_rewards = base_rewards * len(target_rewards)
            if len(base_rewards) == len(target_rewards):
                return [
                    (
                        # take abs in case the reward is (-ppl)
                        (target_rw - base_rw) / abs(base_rw)
                        if self.reward_type.endswith("ratio")
                        else (target_rw - base_rw)
                    )
                    for base_rw, target_rw in zip(base_rewards, target_rewards)
                ]
            else:
                raise Exception(f"bad values: {base_rewards=}, {target_rewards=}")

    def aggregate_rewards(
        self,
        rewards: List[float],
        rm_compo_ranges: Dict[str, List[int]],
        num_rollouts: int,
    ) -> list[float]:
        prefix_base_start, prefix_base_end = rm_compo_ranges["prefix_base"]
        prefix_base_rewards: list[float] = rewards[prefix_base_start:prefix_base_end]
        prefix_target_start, prefix_target_end = rm_compo_ranges["prefix_target"]
        prefix_target_rewards: list[float] = rewards[
            prefix_target_start:prefix_target_end
        ]
        suffix_base_start, suffix_base_end = rm_compo_ranges["suffix_base"]
        suffix_base_rewards: list[float] = rewards[suffix_base_start:suffix_base_end]
        suffix_target_start, suffix_target_end = rm_compo_ranges["suffix_target"]
        suffix_target_rewards: list[float] = rewards[
            suffix_target_start:suffix_target_end
        ]

        prefix_rel_reward_diffs: list[float] = self._cal_reward_diff(
            prefix_base_rewards, prefix_target_rewards, num_rollouts
        )
        fs2_log.info(f"{prefix_rel_reward_diffs=}")
        suffix_rel_reward_diffs: list[float] = self._cal_reward_diff(
            suffix_base_rewards, suffix_target_rewards, num_rollouts
        )
        fs2_log.info(f"{suffix_rel_reward_diffs=}")
        agg_func = self.reward_agg_fn_dict[self.reward_agg]
        return [
            agg_func(r1, r2)
            for r1, r2 in zip(prefix_rel_reward_diffs, suffix_rel_reward_diffs)
        ]

    def _log_human_friendly(
        self,
        B,
        tokenizer,
        rm_vllm_inputs,  # flatten
        all_input_tok_lens,  # flatten
        rewards_batch,
        rm_rollouts,  # flatten
    ):
        assert len(rm_vllm_inputs) == len(all_input_tok_lens) and len(
            all_input_tok_lens
        ) == len(rm_rollouts)
        N = len(rm_vllm_inputs)
        ex_n = N // B
        # group into a 2d list with the first dim equals B
        rm_vllm_inputs_batch = [
            rm_vllm_inputs[i * ex_n : (i + 1) * ex_n] for i in range(B)
        ]
        input_tok_lens_batch = [
            all_input_tok_lens[i * ex_n : (i + 1) * ex_n] for i in range(B)
        ]
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
            fs2_log.info("=" * 6 + f"example {example_i} summary" + "=" * 6)
            for i, (tokens, prefix_len, rm_rollout, reward) in enumerate(
                zip(ex_rm_vllm_inputs, ex_input_tok_lens, ex_rm_rollouts, ex_rewards)
            ):  # individual rollouts
                fs2_log.info("-" * 6 + f"prefix {i}" + "-" * 6)
                fs2_log.info(
                    tokenizer.decode(tokens[:prefix_len], skip_special_tokens=True)
                    + "$"
                )
                if self.n_reason_mask > 0 and i == 0:
                    fs2_log.info("-" * 6 + f"masked tokens groundtruth {i}" + "-" * 6)
                    fs2_log.info(
                        tokenizer.decode(
                            tokens[prefix_len - self.n_reason_mask : prefix_len],
                            skip_special_tokens=True,
                        )
                    )
                fs2_log.info("-" * 6 + f"completion window {i}" + "-" * 6)
                fs2_log.info(
                    "^"
                    + tokenizer.decode(tokens[prefix_len:], skip_special_tokens=True)
                )
                completion_logprobs = rm_rollout.prompt_logprobs[prefix_len:]
                completion_logprobs_vals: List[float] = [
                    list(d.values())[0].logprob for d in completion_logprobs
                ]
                fs2_log.info("-" * 6 + f"completion logprobs {i}" + "-" * 6)
                fs2_log.info(str(completion_logprobs_vals))
                completion_logprobs = [list(d.values())[0] for d in completion_logprobs]
                fs2_log.info("-" * 6 + f"completion token logprobs {i}" + "-" * 6)
                fs2_log.info(str(completion_logprobs))
                fs2_log.info("-" * 6 + f"reward {i}" + "-" * 6)
                fs2_log.info(reward)
            fs2_log.info("-" * 6 + "all rewards in example" + "-" * 6)
            fs2_log.info(ex_rewards)

    def _maybe_log_vllm_policy_outputs(self, vllm_outputs) -> None:
        if fs2_log.is_enabled_for_debug():
            fs2_log.debug("prompt token ids:")
            for i, vllm_output in enumerate(vllm_outputs):
                fs2_log.debug(f"prompt_token_id {i} = {vllm_output.prompt_token_ids}")
                for j, output in enumerate(vllm_output.outputs):
                    fs2_log.debug(f"rollout text {i}.{j} = {output.text}")
                    fs2_log.debug(f"rollout token_ids {i}.{j} = {output.token_ids}")
                    fs2_log.debug(
                        f"rollout finish_reason {i}.{j} = {output.finish_reason}"
                    )
                    fs2_log.debug(f"rollout stop_reason {i}.{j} = {output.stop_reason}")

    def _process_rm_inputs(self, key: str, proc_inputs):
        if self.vllm_proc_name_dict[key] is None:
            rm_vllm_tokens, input_lens = [], []
        else:
            result = self.vllm_input_proc_dict[self.vllm_proc_name_dict[key]](
                proc_inputs
            )  # list of tuples of tokens and input_len
            rm_vllm_tokens, input_lens = zip(*result)
            rm_vllm_tokens: list[List[int]] = list(rm_vllm_tokens)
            input_lens: list[int] = list(input_lens)
        fs2_log.debug(f"{key}_rm_vllm_tokens={rm_vllm_tokens}")
        fs2_log.debug(f"{key}_input_lens={input_lens}")
        return rm_vllm_tokens, input_lens

    @override
    def process_rollouts(
        self,
        vllm_outputs: list[RequestOutput],
        prompt_batch: PromptBatch,
    ):
        all_input_tok_lens = []
        rm_vllm_inputs = []
        batch_text = []
        batch_tokens = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        prefix_batch = prompt_batch.meta_info.get(self.prompt_key)
        completion_batch = prompt_batch.meta_info.get(self.answer_key)
        reason_start_wrap_batch = prompt_batch.meta_info.get(
            self.reason_start_wrap_key, len(prompt_batch.prompts) * ["<think>"]
        )
        reason_end_wrap_batch = prompt_batch.meta_info.get(
            self.reason_end_wrap_key, len(prompt_batch.prompts) * ["</think>"]
        )

        self._maybe_log_vllm_policy_outputs(vllm_outputs)
        fs2_log.debug(f"{completion_batch=}")

        for prefix, vllm_output, completion, reason_start_wrap, reason_end_wrap in zip(
            prefix_batch,
            vllm_outputs,
            completion_batch,
            reason_start_wrap_batch,
            reason_end_wrap_batch,
        ):
            rollouts_text = []
            rollouts_tokens = []
            think_texts = []

            for rollout_output in vllm_output.outputs:  # reasoning in rollouts
                think_texts.append(rollout_output.text)
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)

            proc_inputs: Dict[str, str] = {
                "prefix_w_think_tag": prefix,
                "think_texts": think_texts,
                "suffix": completion,
                "think_end_tag": reason_end_wrap,
                "think_start_tag": reason_start_wrap,
            }
            rm_compo_key_list = [
                "prefix_base",
                "prefix_target",
                "suffix_base",
                "suffix_target",
            ]
            # concat all the rm vllm tokens and record their counts so we can extract later.
            rm_compo_ranges = {}
            all_rm_vllm_tokens = []
            all_input_tok_lens = []
            for key in rm_compo_key_list:
                rm_vllm_tokens, input_lens = self._process_rm_inputs(key, proc_inputs)
                all_rm_vllm_tokens.extend(rm_vllm_tokens)
                start_idx = len(all_input_tok_lens)
                all_input_tok_lens.extend(input_lens)
                end_idx = len(all_input_tok_lens)
                rm_compo_ranges[key] = [start_idx, end_idx]

            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)

            fs2_log.debug(f"{all_rm_vllm_tokens=}")
            fs2_log.debug(f"{all_input_tok_lens=}")
            fs2_log.debug(f"{rm_compo_ranges=}")

            self._dummy_prefix_w_think_tag = proc_inputs["prefix_w_think_tag"]
            self._dummy_suffix = proc_inputs["suffix"]

        rm_sampling_params = {
            "n": 1,
            "max_tokens": 1,
            "prompt_logprobs": 0,
            "detokenize": (
                self.enable_human_friendly_log or fs2_log.is_enabled_for_debug()
            ),
        }

        # if self._gangs.root.rank == 0:
        #     breakpoint()
        # self._gangs.root.barrier()

        rollouts = generate_rollouts(
            all_rm_vllm_tokens,
            dp_gang=self._gangs.dp,
            vllm_model=self.reward_model,
            sampling_params=SamplingParams(**rm_sampling_params),
        )
        fs2_log.debug(f"rm {rollouts=}")

        flat_scores: list[float] = [
            self.extract_scores(rollout.prompt_logprobs, prompt_len=input_len)
            for rollout, input_len in zip(rollouts, all_input_tok_lens)
        ]
        fs2_log.info(f"{flat_scores=}")

        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts
        score_split_sz: int = len(flat_scores) // B
        assert len(flat_scores) % B == 0
        batch_scores: list[list[float]] = [
            flat_scores[i * score_split_sz : (i + 1) * score_split_sz] for i in range(B)
        ]  # B, # of raw scores in batch
        batch_rewards: list[list[float]] = [
            self.aggregate_rewards(scores, rm_compo_ranges, R)
            for scores in batch_scores
        ]
        fs2_log.info(f"{batch_rewards=}")

        if self.enable_human_friendly_log:
            self._log_human_friendly(
                B,
                self.tokenizer,
                all_rm_vllm_tokens,
                all_input_tok_lens,
                batch_rewards,
                rollouts,
            )

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:
        pass
