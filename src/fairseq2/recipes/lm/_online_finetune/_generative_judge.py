# --------------------------------------- Pointwise prompts ---------------------------------------- #

POINTWISE_J1_PROMPT = """
You are given a user question and a response from an AI assistant. Your task is to act as an impartial judge and evaluate how well the response fulfills the user's instructions. You will be shown multiple responses to the same prompt, but only one at a time. Evaluate each response independently. 

Think carefully about how to assess the quality of the response and assign the assistant's response a score 1 if the response is correct, and 0 if not. Enclose the score within <score> and </score> tags.

Format your output like this: 
<think> your_thinking_process </think> 
<score> 0 or 1 </score>

Below are the user's question and the assistant's response:

[User Question]
{instruction}

[The Start of the Assistant's Answer]
{response}
[The End of the Assistant's Answer]
"""

# POINTWISE_J1_PROMPT = """
# You are given a user question and a response from an AI assistant. Your task is to act as an impartial judge and evaluate how well the response fulfills the user's instructions. In order to do so, you will first chunk the response into steps and then assign a score to each chunk or step indepdendently. Assign a score of 1 if a chunk is correct, and 0 if not.

# Chunking Principles:
# 1. Unified purpose: A chunk should serve a single, clear objective. For example: setting up an initial equation, executing a self-contained calculation (like integration by parts), or stating a final/intermediate conclusion. All content within the chunk must directly serve this one core goal.

# 2. Logical Cohesion: All lines within a chunk must form a continuous and uninterrupted logical flow. A new chunk should begin as soon as the focus or purpose of the reasoning shifts.

# 3. Clear Transition: A new chunk must begin when the problem-solving process enters a new phase. This includes transitioning from ”solving for a variable” to "verifying the answer," or inserting an "explanatory side-note" into the main workflow.

# Format rules:
# 1. Use <chunk>...</chunk> to mark the beginning and end of each step. Do not copy the chunk, just summarize it in one sentence.
# 2. After identifying the chunk, provide your evaluation of the chunk within <think>...</think> tags.
# 3. Finally, assign a score of 1 or 0 to each chunk, enclosed within <score>...</score> tags.

# Below are the user's question and the assistant's response:

# [User Question]
# {instruction}

# [The Start of the Assistant's Answer]
# {response}
# [The End of the Assistant's Answer]
# """

# POINTWISE_J1_PROMPT = """
# You are given a user question and a response from an AI assistant. Your task is to act as an impartial judge and evaluate how well the response fulfills the user's instructions. You will be shown multiple responses to the same prompt, but only one at a time. Evaluate each response independently.

# Think carefully about how to assess the quality of the response, and enclose your reasoning within <think> and </think> tags. Your reasoning should include your evaluation criteria, a clear understanding of what an ideal response would look like for this particular question, and a concrete example of such an ideal or reference answer if possible. Then compare the assistant's response to your ideal or reference answer, explaining how it aligns with or deviates from your expectations. Be specific and avoid vague or overly general judgments. Remain as objective as possible.

# Finally, assign the assistant's response a score from 0 to 10, using either an integer or a decimal with up to 0.1 precision. A higher score should indicate a higher-quality response. Enclose the score within <score> and </score> tags.

# Format your output like this:
# <think> your_thinking_process </think>  
# <score> your_score </score>

# Below are the user's question and the assistant's response:

# [User Question]
# {instruction}

# [The Start of the Assistant's Answer]
# {response}
# [The End of the Assistant's Answer]
# """

# Uncomment this for non-verifiable prompt

# POINTWISE_J1_PROMPT = """
# You are given a user question and a response from an AI assistant. Your task is to act as an impartial judge and evaluate how well the response fulfills the user's instructions. You will be shown multiple responses to the same prompt, but only one at a time. Evaluate each response independently. 

# Think carefully about how to assess the quality of the response and assign the assistant's response a score from 0 to 10, using either an integer or a decimal with up to 0.1 precision. A higher score should indicate a higher-quality response. Enclose the score within <score> and </score> tags.

# Format your output like this: 
# <think> your_thinking_process </think> 
# <score> your_score </score>

# Below are the user's question and the assistant's response:

# [User Question]
# {instruction}

# [The Start of the Assistant's Answer]
# {response}
# [The End of the Assistant's Answer]
# """


POINTWISE_J1_PROMPT_WITH_REF_ANSWER = """
You are given a user question, a reference answer and a response from an AI assistant. Your task is to act as an impartial judge and evaluate how well the response fulfills the user's instructions. You will be shown multiple responses to the same prompt, but only one at a time. Evaluate each response independently. 

Think carefully about how to assess the quality of the response and assign the assistant's response a score 1 if the response is correct, and 0 if not. Enclose the score within <score> and </score> tags.

Format your output like this: 
<think> your_thinking_process </think> 
<score> 0 or 1 </score>

Below are the user's question, reference answer and the assistant's response:

[User Question]
{instruction}

[Reference Answer]
{reference_answer}

[The Start of the Assistant's Answer]
{response}
[The End of the Assistant's Answer]
"""

# --------------------------------------- Pairwise prompts ---------------------------------------- #

PAIRWISE_WITH_SCORES_J1_PROMPT = """
You are given a user question and two responses from two AI assistants. Your task is to act as an impartial judge and evaluate which response better follows the user's instructions and provides a higher-quality answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

Think carefully about how to assess the quality of the responses and assign each response a score 1 if the response is correct, and 0 if not. Enclose the scores within the tags <score_A> </score_A>, and <score_B> </score_B>.

Format your output like this:
<think> your_thinking_process </think>
<score_A> 0 or 1 </score_A> <score_B> 0 or 1 </score_B>

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

# Uncomment this for non-verifiable prompt

# PAIRWISE_WITH_SCORES_J1_PROMPT = """
# You are given a user question and two responses from two AI assistants. Your task is to act as an impartial judge and evaluate which response better follows the user's instructions and provides a higher-quality answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

# Think carefully about how to assess the quality of the responses and assign each response a score from 0 to 10, using either an integer or a decimal with up to 0.1 precision, with a higher score indicating a higher-quality response that better satisfies the criteria. Enclose the scores within the tags <score_A> </score_A>, and <score_B> </score_B>.

# Format your output like this:
# <think> your_thinking_process </think>
# <score_A> your_score_a </score_A> <score_B> your_score_b </score_B>

# Below are the user's question and the two responses:

# [User Question]
# {instruction}

# [The Start of Assistant A's Answer]
# {response_A}
# [The End of Assistant A's Answer]

# [The Start of Assistant B's Answer]
# {response_B}
# [The End of Assistant B's Answer]
# """

PAIRWISE_WITH_SCORES_J1_PROMPT_WITH_REF_ANSWER = """
You are given a user question, two responses from two AI assistants and the parsed version of the responses, and a reference answer. Your task is to act as an impartial judge and evaluate which response better follows the user's instructions and provides a higher-quality answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

Think carefully about how to assess the quality of the responses and finally, utilize the reference answer for your judgement. Note that the parsed version of the responses are automatically extracted and may contain errors, therefore you should primarily rely on the original responses for your judgement.
Finally, assign each response a score 1 if the response is correct, and 0 if not. Enclose the scores within the tags <score_A> </score_A>, and <score_B> </score_B>.

Format your output like this:
<think> your_thinking_process </think>
<score_A> 0 or 1 </score_A> <score_B> 0 or 1 </score_B>

Below are the user's question, two responses and the parsed versions of the responses, and the reference answer:

[User Question]
{instruction}

[The Start of Assistant A's Answer]
{response_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_B}
[The End of Assistant B's Answer]

[The Parsed Version of Assistant A's Answer]
{parsed_response_A}

[The Parsed Version of Assistant B's Answer]
{parsed_response_B}

[Reference Answer]
{reference_answer}
"""


# --------------------------------------- K-wise prompts ---------------------------------------- #

KWISE_WITH_SCORES_J1_PROMPT = """
You are given a user question and {k} responses from {k} AI assistants. Your task is to act as an impartial judge and evaluate which response better follows the user's instructions and provides a higher-quality answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

Think carefully about how to assess the quality of the responses and finally, assign each response a score from 0 to 10, using either an integer or a decimal with up to 0.1 precision, with a higher score indicating a higher-quality response that better satisfies the criteria. Enclose the scores within the tags <score_assistant_1> </score_assistant_1>, <score_assistant_2> </score_assistant_2> and so on.

Format your output like this:
<think> your_thinking_process </think>
<score_assistant_1> your_score_1 </score_assistant_1> 
<score_assistant_2> your_score_2 </score_assistant_2>
<score_assistant_3> your_score_3 </score_assistant_3>
...

Below are the user's question and the responses:

[User Question]
{instruction}

{responses}
"""

KWISE_WITH_SCORES_J1_PROMPT_WITH_REF_ANSWER = """
You are given a user question, a reference answer, and {k} responses with the parsed versions from AI assistants. Your task is to act as an impartial judge and evaluate which response better follows the user's instructions and provides a higher-quality answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

Think carefully about how to assess the quality of the responses and finally, utilize the reference answer for your judgement. Note that the parsed version of the responses are automatically extracted and may contain errors, therefore you should primarily rely on the original responses for your judgement.
Finally, assign each response a score 1 if the response is correct, and 0 if not. Enclose the scores within the tags <score_assistant_1> </score_assistant_1>, <score_assistant_2> </score_assistant_2> and so on.

Format your output like this:
<think> your_thinking_process </think>
<score_assistant_1> 0 or 1 </score_assistant_1> 
<score_assistant_2> 0 or 1 </score_assistant_2>
<score_assistant_3> 0 or 1 </score_assistant_3>
...

Below are the user's question, reference answer, responses and the parsed versions of the responses:

[User Question]
{instruction}

[Reference Answer]
{reference_answer}

{responses}

{parsed_responses}
"""

PRINCIPIA_JUDGE_PROMPT = """### Question: {instruction}

### Ground Truth Answer: {ground_truth}

### Candidate: {candidate}

### Guidelines: For the above question, please verify if the candidate is equivalent with the ground truth answer or not.
DO NOT ATTEMPT TO SOLVE the question by yourself; instead focus on checking if the two candidates are equivalent.
If the two candidates are equivalent, output "Final Judgment: Yes <End of Judgment>". If not, output "Final Judgment: No <End of Judgment>". Most importantly, DO NOT MAKE a judgment first. Instead, first reason about whether the candidates are equivalent or not based on the specified rules above (read through all of them, not only one), and then output the final judgment.

### Reasoning:
"""



import re
from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import override

from fairseq2.logging import log


class JudgmentExtractorHandler(ABC):
    @abstractmethod
    def create(self, tokenizer):
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]:
        ...


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
    def prompt(self) -> str:
        ...

    @abstractmethod
    def format_prompt(self, prompt_text, **kwargs: Any) -> str:
        ...

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
    def extract(self, generation) -> float | str:
        ...

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
    def aggregate(self, judgments) -> float | str:
        ...

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
    def create(self, tokenizer):
        return GeneralVerifierExtractor(tokenizer)

    @property
    @override
    def name(self):
        return "general_verifier_extractor"

    @property
    @override
    def config_kls(self):
        return None


class GeneralVerifierExtractor(JudgmentExtractor):
    def __init__(self, tokenizer):
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
    
class PrincipiaExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self, tokenizer):
        return PrincipiaExtractor(tokenizer)

    @property
    @override
    def name(self):
        return "principia_extractor"

    @property
    @override
    def config_kls(self):
        return None


class PrincipiaExtractor(JudgmentExtractor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @override
    def prompt(self):
        return PRINCIPIA_JUDGE_PROMPT

    @override
    def format_prompt(self, prompt_text, rollout_text, reference_answer):
        boxed_answer = self.extract_boxed_answer(rollout_text)
        prompt_template = self.prompt()
        content = (
            prompt_template.format(instruction=prompt_text, ground_truth=reference_answer, candidate=boxed_answer)
        )

        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self.tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        log.info(f"Judge input = {chat_str}")
        return chat_str
    
    def extract_boxed_answer(self, response):
        """
        Extract content from the last \\boxed{} in the response.
        Handles nested braces correctly and returns only the last boxed item.
        """
        pattern = r"\\boxed\s*\{"
        all_answers = []

        # Find all \boxed{} occurrences
        for match in re.finditer(pattern, response):
            start_idx = match.end()
            brace_count = 1
            idx = start_idx

            while idx < len(response) and brace_count > 0:
                if response[idx] == "{":
                    brace_count += 1
                elif response[idx] == "}":
                    brace_count -= 1
                idx += 1

            if brace_count == 0:
                all_answers.append(response[start_idx : idx - 1])

        # Return the last boxed answer, or None if no valid boxed content found
        return all_answers[-1] if all_answers else ""

    @override
    def extract(self, generation):
        pattern = r"Final\s+Judgment[:\s]*\s*(Yes|No)"
        match = re.search(pattern, generation, re.IGNORECASE)

        if match:
            judgment = match.group(1).capitalize()
            return 1.0 if judgment == "Yes" else 0.0

        return 0.0

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
    def create(self, tokenizer):
        return J1PointwiseExtractor(tokenizer)

    @property
    @override
    def name(self):
        return "j1_pointwise_extractor"

    @property
    @override
    def config_kls(self):
        return None


class J1PointwiseExtractor(JudgmentExtractor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @override
    def prompt(self, reference_answer):
        return (
            POINTWISE_J1_PROMPT
            if reference_answer is None
            else POINTWISE_J1_PROMPT_WITH_REF_ANSWER
        )

    @override
    def format_prompt(self, prompt_text, rollout_text, reference_answer):
        prompt_template = self.prompt(reference_answer)
        content = (
            prompt_template.format(instruction=prompt_text, response=rollout_text)
            if reference_answer is None
            else prompt_template.format(
                instruction=prompt_text,
                reference_answer=reference_answer,
                response=rollout_text,
            )
        )

        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self.tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        # log.info(f"Judge input = {chat_str}")
        return chat_str

    @override
    def extract(self, generation):
        matches = re.findall(
            r"<score>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score>", generation
        )
        avg_score = 0.0
        if matches:
            for match in matches:
                avg_score += float(match.strip())
            return round(avg_score / len(matches), 4)
        else:
            return 0.0

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
    def create(self, tokenizer):
        return J1PairwiseScoreExtractor(tokenizer)

    @property
    @override
    def name(self):
        return "j1_pairwise_score_extractor"

    @property
    @override
    def config_kls(self):
        return None


class J1PairwiseScoreExtractor(JudgmentExtractor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
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
    def prompt(self, reference_answer):
        return (
            PAIRWISE_WITH_SCORES_J1_PROMPT
            if reference_answer is None
            else PAIRWISE_WITH_SCORES_J1_PROMPT_WITH_REF_ANSWER
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
    def format_prompt(
        self, prompt_text, rollout_A_text, rollout_B_text, reference_answer
    ):
        prompt_template = self.prompt(reference_answer)
        content = (
            prompt_template.format(
                instruction=prompt_text,
                response_A=rollout_A_text,
                response_B=rollout_B_text,
            )
            if reference_answer is None
            else prompt_template.format(
                instruction=prompt_text,
                response_A=rollout_A_text,
                response_B=rollout_B_text,
                parsed_response_A=self.get_preferred_index(self.parse(rollout_A_text, self.student_extraction_config)),
                parsed_response_B=self.get_preferred_index(self.parse(rollout_B_text, self.student_extraction_config)),
                reference_answer=reference_answer,
            )
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


class J1KwiseScoreExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self, tokenizer, k):
        return J1KwiseScoreExtractor(tokenizer, k)

    @property
    @override
    def name(self):
        return "j1_kwise_score_extractor"

    @property
    @override
    def config_kls(self):
        return None


class J1KwiseScoreExtractor(JudgmentExtractor):
    def __init__(self, tokenizer, k):
        self.tokenizer = tokenizer
        self.k = k
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
    def prompt(self, reference_answer):
        return (
            KWISE_WITH_SCORES_J1_PROMPT
            if reference_answer is None
            else KWISE_WITH_SCORES_J1_PROMPT_WITH_REF_ANSWER
        )

    @override
    def format_prompt(self, prompt_text, rollouts, reference_answer):
        prompt_template = self.prompt(reference_answer)
        content = (
            prompt_template.format(
                k=self.k, 
                instruction=prompt_text, 
                responses="".join([f"[Start of Assistant {assistant_id+1}'s Answer]\n{rollout}\n[End of Assistant {assistant_id+1}'s Answer]\n\n" for assistant_id, rollout in enumerate(rollouts)])
            )
            if reference_answer is None
            else prompt_template.format(
                k=self.k,
                instruction=prompt_text,
                responses="".join([f"[Start of Assistant {assistant_id+1}'s Answer]\n{rollout}\n[End of Assistant {assistant_id+1}'s Answer]\n\n" for assistant_id, rollout in enumerate(rollouts)]),
                parsed_responses="".join([f"[The Parsed Version of Assistant {assistant_id+1}'s Answer]\n{self.get_preferred_index(self.parse(rollout, self.student_extraction_config))}\n\n" for assistant_id, rollout in enumerate(rollouts)]),
                reference_answer=reference_answer,
            )
        )

        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self.tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        return chat_str

    @override
    def extract(self, generation):
        scores = []
        for i in range(self.k):
            score_matches = re.findall(
                rf"<score_assistant_{i+1}>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score_assistant_{i+1}>",
                generation,
            )
            if score_matches:
                scores.append(float(score_matches[-1].strip()))
            else:
                scores.append(0.0)

        return scores

    @override
    def aggregate(self, judgments):
        avg_score = [0.0] * self.k
        for scores in judgments:
            for i, score in enumerate(scores):
                avg_score[i] += score

        avg_score = [round(avg_score[i] / len(judgments), 4) for i in range(self.k)]
        return avg_score
