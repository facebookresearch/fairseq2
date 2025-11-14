POINTWISE_J1_PROMPT = """
You are given a user question and a response from an AI assistant. Your task is to act as an impartial judge and evaluate how well the response fulfills the user's instructions. You will be shown multiple responses to the same prompt, but only one at a time. Evaluate each response independently.

Think carefully about how to assess the quality of the response, and enclose your reasoning within <think> and </think> tags. Your reasoning should include your evaluation criteria, a clear understanding of what an ideal response would look like for this particular question, and a concrete example of such an ideal or reference answer if possible. Then compare the assistant's response to your ideal or reference answer, explaining how it aligns with or deviates from your expectations. Be specific and avoid vague or overly general judgments. Remain as objective as possible.

Finally, assign the assistant's response a score from 0 to 10, using either an integer or a decimal with up to 0.1 precision. A higher score should indicate a higher-quality response. Enclose the score within <score> and </score> tags.

Format your output like this:
<think> your_thinking_process </think>
<score> your_score </score>

Below are the user's question and the assistant's response:

[User Question]
{instruction}

[The Start of the Assistant's Answer]
{response}
[The End of the Assistant's Answer]
"""

PAIRWISE_J1_PROMPT = """
You are given a user question and two responses from two AI assistants. Your task is to act as an impartial judge and evaluate which response better follows the user's instructions and provides a higher-quality answer.

First, provide your reasoning within <think> and </think> tags. This should include your evaluation criteria for a high-quality response, a detailed comparison of the two responses, and when helpful, a reference answer as part of your evaluation. Be explicit in your thought process, referencing your criteria and explaining how each response aligns with or deviates from them.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

Finally, provide your verdict within <answer> and </answer> tags, strictly following this format:
- <answer> [[A]] </answer> if Assistant A is better
- <answer> [[B]] </answer> if Assistant B is better

Below are the user's question and the two responses:

[User Question]
{instruction}

[The Start of Assistant A's Answer]
{response_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_B}
[The End of Assistant B's Answer]
"""

PAIRWISE_WITH_SCORES_J1_PROMPT = """
You are given a user question and two responses from two AI assistants. Your task is to act as an impartial judge and evaluate which response better follows the user's instructions and provides a higher-quality answer.

First, provide your reasoning within <think> and </think> tags. This should include your evaluation criteria for a high-quality response, a detailed comparison of the two responses, and when helpful, a reference answer as part of your evaluation. Be explicit in your thought process, referencing your criteria and explaining how each response aligns with or deviates from them.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

Finally, assign the assistant's response a score from 0 to 10, using either an integer or a decimal with up to 0.1 precision, with a higher score indicating a higher-quality response that better satisfies the criteria. Enclose the scores within the tags <score_A> </score_A>, and <score_B> </score_B>.

Format your output like this:
<think> your_thinking_process </think>
<score_A> your_score_a </score_A> <score_B> your_score_b </score_B>

Below are the user's question and the two responses:

[User Question]
{instruction}

[The Start of Assistant A's Answer]
{response_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_B}
[The End of Assistant B's Answer]
"""

SELF_AUGMENTING_PROMPT = """
You are given a ground truth text, and a generated text from an AI assistant. Your task is to act as an impartial judge and evaluate how well the response matches the ground truth text. It doesn't have to match word for word, but it should be very similar.

Think carefully about how to assess how well the generated text matches the ground truth. Your reasoning should include your evaluation criteria.

Finally, assign the assistant's generation a binary score, either 0 or 1. A 0 indicates that the generated text does not match the ground truth text, and a 1 indicates that it matches well.

Format your score as \\boxed{{SCORE}} where SCORE is either 0 or 1.

Below are the ground truth text and the assistant's Generation:

[Start of Ground Truth Text]
{ground_truth}
[End of Ground Truth Text]

[Start of Assistant's Generation]
{generation}
[End of Assistant's Generation]
"""


import re
from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import override

from fairseq2.logging import log


class JudgmentExtractorHandler(ABC):
    @abstractmethod
    def create(self): ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


"""
All judgment extractors are expected to:
(1) define their judgment prompt
(2) implement their judgment (i.e., scores, preferences, etc) extraction logic from the CoTs
(3) implement their aggregation logic over judgments, if sampling multiple CoTs
"""


class JudgmentExtractor(ABC):
    """
    This class defines the interface for extracting judgments from generative models,
    including formatting prompts for the reward model, extracting scalar scores from
    model responses, and aggregating multiple judgments into a single value.
    """

    @abstractmethod
    def prompt(self) -> str: ...

    @abstractmethod
    def format_prompt(self, prompt_text, **kwargs: Any) -> str: ...

    """
    Format the prompt text and additional arguments into a string suitable for input to the reward model.
    This method is responsible for formatting the question and responses as input to the reward model.
    Args:
        prompt_text (str): The main prompt or question text to be formatted.
        **kwargs (Any): Additional keyword arguments that may be required for formatting such as rollout_text, reference_answer, etc.
    Returns:
        str: The formatted prompt string ready for the reward model.
    """

    @abstractmethod
    def extract(self, generation) -> float | str: ...

    """
    Extract the final scalar reward score from the model's response.
    This should be implemented to process the given `generation`
    and return either a float representing the reward score or a string with
    additional information.

    Args:
        generation: The model's generated response to be evaluated.

    Returns:
        float | str: The extracted scalar reward score or a string with details.

    Note:
        This method is intended for extracting the final scalar reward score from the model's response.
    """

    @abstractmethod
    def aggregate(self, judgments) -> float | str: ...

    """
    Aggregate multiple responses (judgments) from the reward model into a single value.
    This should combine the results of several model outputs (e.g., scores or preferences)
    into a final scalar or summary value, such as an average score or majority preference.

    Args:
        judgments: A list of individual judgments (e.g., scores or preferences) to aggregate.

    Returns:
        float | str: The aggregated result, such as an average score or consensus preference.
    """


class GeneralVerifierExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self):
        return GeneralVerifierExtractor()

    @property
    @override
    def name(self):
        return "general_verifier_extractor"

    @property
    @override
    def config_kls(self):
        return None


class GeneralVerifierExtractor(JudgmentExtractor):
    def __init__(self):
        try:
            from math_verify import parse
            from math_verify.parser import (
                ExprExtractionConfig,
                LatexExtractionConfig,
                NormalizationConfig,
            )
        except ImportError:
            raise ImportError(
                "install mathverify from https://github.com/huggingface/Math-Verify"
            )

        self.student_extraction_config = (
            LatexExtractionConfig(boxed_match_priority=0),
        )
        self.parse = parse

    @override
    def prompt(self):
        raise NotImplementedError(
            "Using the string provided by the general verifier code in format_prompt instead"
        )

    def get_preferred_index(self, lst):
        """
        math_verify parse returns a list of parsed answers, we want want the item at idex 1, which is a string
        """
        if len(lst) > 1:
            return lst[1]
        elif len(lst) == 1:
            return lst[0]
        else:
            return "None"

    @override
    def format_prompt(self, prompt_text, rollout_text, reference_answer):

        student_answer_list = self.parse(rollout_text, self.student_extraction_config)
        student_answer = self.get_preferred_index(student_answer_list)

        prompt = (
            f"User: ### Question: {prompt_text}\n\n"
            f"### Ground Truth Answer: {reference_answer}\n\n"
            f"### Student Answer: {student_answer}\n\n"
            "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
            "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
            'If the student\'s answer is correct, output "Final Decision: Yes". If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
        )

        return prompt

    @override
    def extract(self, generation):
        if "Final Decision: Yes" in generation:
            return 1.0
        else:
            return 0.0

    @override
    def aggregate(self, judgments):
        avg_score = 0.0
        for score in judgments:
            avg_score += score

        return round(avg_score / len(judgments), 4)


class SelfAugmentingExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self):
        return SelfAugmentingExtractor()

    @property
    @override
    def name(self):
        return "self_augmenting_extractor"

    @property
    @override
    def config_kls(self):
        return None


class SelfAugmentingExtractor(JudgmentExtractor):
    def __init__(
        self,
    ):
        pass

    @override
    def prompt(self):
        return SELF_AUGMENTING_PROMPT


    def remove_think_tags(self, rollout_text):
        tag = "</think>"
        count = rollout_text.count(tag)
        if count == 1:
            # Find the position after the tag and return everything after it
            index = rollout_text.find(tag) + len(tag)
            return rollout_text[index:]
        else:
            return "" # set rollout to empty string if it doesn't contain thought or has multiple

    @override
    def format_prompt(self, tokenizer, prompt_text, rollout_text, reference_answer, dp_gangs):
        # if dp_gangs.rank == 0
        #     breakpoint()
        # dp_gangs.root.barrier()

        rollout_text = self.remove_think_tags(rollout_text)

        content = self.prompt().format(ground_truth=reference_answer, generation=rollout_text)

        # log.info(f"Judge prompt = {content}")
        wrapped_text = [{"role": "user", "content": content}]
        chat_str = tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        return chat_str

    @override
    def extract(self, generation):
        # pattern = r'\\boxed\{(-?\d+)\}'
        pattern = r'\\boxed\{([01])\}'
        match = re.search(pattern, generation)
        if match:
            score = float(match.group(1))
        else:
            score = 0.0
        return score

    @override
    def aggregate(self, judgments):
        avg_score = 0.0
        for score in judgments:
            avg_score += score

        return round(avg_score / len(judgments), 4)

class J1PointwiseExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self):
        return J1PointwiseExtractor()

    @property
    @override
    def name(self):
        return "j1_pointwise_extractor"

    @property
    @override
    def config_kls(self):
        return None


class J1PointwiseExtractor(JudgmentExtractor):
    def __init__(
        self,
    ):
        pass

    @override
    def prompt(self):
        return POINTWISE_J1_PROMPT

    @override
    def format_prompt(self, prompt_text, rollout_text, reference_answer):
        content = self.prompt().format(instruction=prompt_text, response=rollout_text)
        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self.tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        return chat_str

    @override
    def extract(self, generation):
        matches = re.findall(
            r"<score>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score>", generation
        )
        if matches and float(matches[-1].strip()) > 10.0:
            log.info(f"Judge output = {generation}")
        return float(matches[-1].strip()) if matches else 0.0

    @override
    def aggregate(self, judgments):
        avg_score = 0.0
        for score in judgments:
            avg_score += score

        return round(avg_score / len(judgments), 4)


class J1PairwiseScoreExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self):
        return J1PairwiseScoreExtractor()

    @property
    @override
    def name(self):
        return "j1_pairwise_score_extractor"

    @property
    @override
    def config_kls(self):
        return None


class J1PairwiseScoreExtractor(JudgmentExtractor):
    def __init__(self):
        pass

    @override
    def prompt(self):
        return PAIRWISE_WITH_SCORES_J1_PROMPT

    @override
    def format_prompt(self, prompt_text, rollout_A_text, rollout_B_text):
        content = self.prompt().format(
            instruction=prompt_text,
            response_A=rollout_A_text,
            response_B=rollout_B_text,
        )
        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self.tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        return chat_str

    @override
    def extract(self, generation):
        score_a_matches = re.findall(
            r"<score_A>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score_A>", generation
        )
        score_b_matches = re.findall(
            r"<score_B>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score_B>", generation
        )

        if score_a_matches and score_b_matches:
            score_a = score_a_matches[-1]
            score_b = score_b_matches[-1]
            if float(score_a.strip()) > 10.0 or float(score_b.strip()) > 10.0:
                log.info(f"Judge output = {generation}")
            return (float(score_a.strip()), float(score_b.strip()))
        else:
            return (0.0, 0.0)

    @override
    def aggregate(self, judgments):
        avg_score = (0.0, 0.0)
        for score in judgments:
            avg_score = (avg_score[0] + score[0], avg_score[1] + score[1])

        return (
            round(avg_score[0] / len(judgments), 4),
            round(avg_score[1] / len(judgments), 4),
        )


class J1PairwisePreferenceExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self):
        return J1PairwisePreferenceExtractor()

    @property
    @override
    def name(self):
        return "j1_pairwise_preference_extractor"

    @property
    @override
    def config_kls(self):
        return None


class J1PairwisePreferenceExtractor(JudgmentExtractor):
    def __init__(self):
        pass

    @override
    def prompt(self):
        return PAIRWISE_J1_PROMPT

    @override
    def extract(self, generation):
        matches = list(
            re.findall(r"<answer>\s*\[\[(A|B)\]\]\s*</answer>", generation.strip())
        )

        return matches[-1].strip() if matches else None

    @override
    def aggregate(self, judgments):
        pass
