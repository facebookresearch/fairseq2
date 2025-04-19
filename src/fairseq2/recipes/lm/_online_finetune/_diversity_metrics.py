import string as string_lib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import gzip
import torch
from fairseq2.recipes.lm._online_finetune._rewards import extract_logprobs


def get_compression_ratio(strings):

    flattened_generation = " ".join(strings)
    original_byte_size = len(bytes(flattened_generation, "UTF-8"))
    compressed_bytes_size = len(gzip.compress(bytes(flattened_generation, "UTF-8")))

    cr = compressed_bytes_size / original_byte_size
    cr_tensor = torch.Tensor([cr])
    return cr_tensor


def get_self_bleu_score(strings):
    # Create a translation table to remove punctuation
    translator = str.maketrans("", "", string_lib.punctuation)

    # Preprocess the strings: convert to lowercase and remove punctuation
    cleaned_strings = [s.lower().translate(translator) for s in strings]

    # Tokenize the cleaned strings into lists of words
    tokenized_strings = [s.split() for s in cleaned_strings]

    # Initialize a dictionary to store BLEU scores
    bleu_scores = []

    # Calculate BLEU scores for all pairs of strings
    for i in range(len(tokenized_strings)):
        for j in range(i + 1, len(tokenized_strings)):
            # Use smoothing to handle cases where there are no n-grams in common
            smoothie = SmoothingFunction().method4
            bleu = sentence_bleu(
                [tokenized_strings[i]],
                tokenized_strings[j],
                smoothing_function=smoothie,
            )

            # Store the BLEU score
            bleu_scores.append(bleu)

    mean_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    mean_bleu_score_tensor = torch.Tensor([mean_bleu_score])
    return mean_bleu_score_tensor


def get_unique_1grams(strings):

    # Initialize an empty set to store unique 1-grams
    unique_words = set()
    total_words = 0

    # Create a translation table to remove punctuation
    translator = str.maketrans("", "", string_lib.punctuation)

    # Iterate over each string in the list
    for string in strings:
        # Convert the string to lowercase and remove punctuation
        cleaned_string = string.lower().translate(translator)

        # Split the cleaned string into words (1-grams) and update the set
        words = cleaned_string.split()
        total_words += len(words)
        unique_words.update(words)

    # Return the set of unique 1-grams
    num_unique_1grams = len(unique_words)
    num_unique_1grams_norm = len(unique_words) / total_words if total_words > 0 else 0
    num_unique_1grams_tensor = torch.Tensor([num_unique_1grams])
    num_unique_1grams_norm = torch.Tensor([num_unique_1grams_norm])
    return num_unique_1grams_tensor, num_unique_1grams_norm


def get_entropy(rollouts):
    batch_sum_logprobs = []
    batch_sum_logprobs_per_tok = []
    for rollout_idx in range(len(rollouts[0].outputs)):
        logprobs = extract_logprobs(rollouts[0].outputs[rollout_idx].logprobs)

        sum_logprobs = -sum(logprobs)
        sum_logprobs_per_tok = -sum(logprobs) / len(logprobs)

        batch_sum_logprobs.append(sum_logprobs)
        batch_sum_logprobs_per_tok.append(sum_logprobs_per_tok)

    entropy = sum(batch_sum_logprobs) / len(batch_sum_logprobs)
    entropy_norm = sum(batch_sum_logprobs_per_tok) / len(batch_sum_logprobs_per_tok)

    return entropy, entropy_norm
