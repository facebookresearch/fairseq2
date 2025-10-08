import contextlib
import io
from ..dataset import PromptBatch
from .utils import prepare_preference_batch_random_pair

from .utils import Reward
from vllm import RequestOutput

@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield

class MathVerifyVerifier(Reward):
    def __init__(self, answer_key, prompt_key, reward_name, gangs):
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

    def process_rollouts(
        self,
        vllm_outputs: list[RequestOutput],
        prompt_batch: PromptBatch,
    ):
        batch_text = []
        batch_tokens = []
        batch_rewards = []

        reference_answers: list[object] = prompt_batch.meta_info.get(self.answer_key)

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
    ):
        """
        reward specific preference batch logic is here since math verify is 1/0 reward meaning that preference can be undefined or ambiguous
        """

        reward_output = self.process_rollouts(rollouts, prompt_batch)

        result = prepare_preference_batch_random_pair(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        return result