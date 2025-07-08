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

from abc import ABC, abstractmethod
from typing_extensions import override
from fairseq2.logging import log
import re


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
    @abstractmethod
    def prompt(self) -> str: ...

    @abstractmethod
    def extract(self, generation) -> float | str: ...

    @abstractmethod
    def aggregate(self, judgments) -> float | str: ...


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
    def __init__(self):
        pass

    @override
    def prompt(self):
        return POINTWISE_J1_PROMPT

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
