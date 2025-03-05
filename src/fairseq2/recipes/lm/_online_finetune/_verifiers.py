# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import override
from typing import Any, List

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.gang import Gangs
from fairseq2.recipes.model import Model
from fairseq2.recipes.trainer import TrainUnit
from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput
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
    def __call__(self, vllm_outputs): ...


class GSM8kVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, recipe_config):
        return GSM8kVerifier()
    
    @property
    @override
    def name(self):
        return "gsm8k_verifier"
    
    @property
    @override
    def config_kls(self):
        return None


class GSM8kVerifier(VLLMOutputReward):
    def __init__(self):
        self.answer_re = re.compile(r"#### (\-?[0-9\.\,]+)")  # regexp from original gsm8k to extract formatted answer
        self.invalid_answer = "[invalid]"

    def extract_answer(self, completion: str):
        match = self.answer_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.invalid_answer

    @override
    def __call__(self, vllm_outputs: List[RequestOutput], reference_answers: List[str]):
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