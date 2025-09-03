###################################
# this file splits input tokens into prefix and completion based on entropy so
# that our model can learn to reason on more difficult tokens.
###################################

import json
import os
import random
import re
from collections import Counter
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

ENTROPY_THRESHOLD: float = (
    1.5  # TODO(lidli): consider upsampling hard tokens rather than doing a filtering.
)
INPUT_DIR = "/fsx-ram/lidli/datasets/natural_reasoning_data_extracted_w_token_entropy"
INPUT_FILE_REGEX = r"data.chunk.\d+.jsonl"
OUPUT_DIR = f"/fsx-ram/lidli/datasets/natural_reasoning_data_extracted_perplexity_scores_1B_split_for_rl_entropy_{str(ENTROPY_THRESHOLD).replace('.','_')}"


# constants
COMPLETION_FIELD: str = "completion"
PREFIX_FIELD: str = "prefix"
TOKENS_FIELD: str = "tokens"
PREFIX_END_FIELD: str = "prefix_token_end"
TOKENIZER_MODEL_ID: str = "/fsx-ram/shared/Llama-3.2-1B"
BOS_TOKEN: str = "<|begin_of_text|>"
# debugging option
DOC_LIMIT: Optional[int] = None


def write_sorted_table(text_f, data, out_dir, plot_name):
    bins, counts = zip(*sorted(data.items()))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{plot_name}.png")
    plt.figure(figsize=(10, 5))
    plt.bar(bins, counts, width=0.09, edgecolor="black", align="center")
    plt.xlabel("val")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(fig_path)  # saves as file instead of showing interactively
    print(f"figure in {fig_path}")

    for k, v in sorted(data.items()):
        text_f.write(f"{k}: {v}")
        text_f.write("\n")


def write_data(f_dir: str, f_name: str, data: List[Any]):
    path = os.path.join(f_dir, f_name)
    print(f"writing the data to {path}")
    with open(path, "w", encoding="utf-8") as fout:
        for output in data:
            json.dump(output, fout)
            fout.write("\n")


# filter all the input files in input dir.
all_files_input_dir = os.listdir(INPUT_DIR)
in_files = [
    file
    for file in all_files_input_dir
    if re.fullmatch(INPUT_FILE_REGEX, file) is not None
]
in_files.sort()
print("=" * 20)
print(f"all input files: {in_files}")

os.makedirs(OUPUT_DIR, exist_ok=True)
print(f"{OUPUT_DIR=}")

print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, use_fast=False)
print("finished loadding tokenizer")

eligible_split_ratios = []
entropy_counter = Counter()
for in_file in tqdm(in_files, desc="files"):
    n_skipped_docs: int = 0
    n_total_docs: int = 0
    print(f"Processing {in_file}")
    with open(os.path.join(INPUT_DIR, in_file), "r", encoding="utf-8") as in_f:
        docs = in_f.readlines()

    outputs = []

    if DOC_LIMIT is not None:
        print(f"=========DEBUGGING WITH {DOC_LIMIT=}=========")
        print(f"{DOC_LIMIT=}!")
        docs = docs[: min(DOC_LIMIT, len(docs))]
    for doc in tqdm(docs, desc="docs"):
        doc_tokens, doc_logprobs = [], []
        doc_sentences = []
        doc_data = json.loads(doc)

        # extract and validate fields
        prompt_token_ids = doc_data["vllm_output"]["prompt_token_ids"]
        prompt_entropies = doc_data["vllm_output"]["prompt_entropies"]
        if prompt_entropies[0] is None:
            prompt_entropies = prompt_entropies[1:]
        entropy_counter.update(round(entropy, 1) for entropy in prompt_entropies)
        assert len(prompt_token_ids) == len(prompt_entropies) + 1

        entropy_i_list = [
            i
            for i, entropy in enumerate(prompt_entropies)
            if entropy >= ENTROPY_THRESHOLD
        ]
        curr_eligible_split_ratio = len(entropy_i_list) / len(prompt_token_ids)
        eligible_split_ratios.append(curr_eligible_split_ratio)
        # print(curr_eligible_split_ratio)
        if entropy_i_list:
            entropy_split_i = random.choice(entropy_i_list)
        else:
            entropy_split_i = random.randrange(len(prompt_entropies))
        output = {}
        # note entropy index is 1 + token index since we removed the none in the beginning
        entropy_split_i += 1
        output[PREFIX_FIELD] = (
            tokenizer.decode(
                prompt_token_ids[:entropy_split_i],
                add_special_tokens=False,
            ).removeprefix(BOS_TOKEN)
            + "<think>"
        )
        output[COMPLETION_FIELD] = tokenizer.decode(
            prompt_token_ids[entropy_split_i:], add_special_tokens=False
        )
        # output[TOKENS_FIELD] = prompt_token_ids
        # output[PREFIX_END_FIELD] = entropy_split_i
        n_total_docs += 1
        outputs.append(output)

    # print summary
    print(f"summary: {in_file=}, {n_skipped_docs=}, {n_total_docs=}, {len(outputs)=}")
    write_data(OUPUT_DIR, in_file, outputs)

# statistics
hist, bin_edges = np.histogram(eligible_split_ratios, bins=10)
valid_ratio_distribution = {
    f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}": int(hist[i]) for i in range(len(hist))
}


stat_dir = os.path.join(OUPUT_DIR, "stats")
stats_path = os.path.join(OUPUT_DIR, "entropy_stats.txt")
with open(stats_path, "w") as f:
    f.write("=" * 20 + "\n")
    f.write(f"Valid Token Frequency distribution under threshold {ENTROPY_THRESHOLD}\n")
    write_sorted_table(f, entropy_counter, stat_dir, "entropy_distribution")
    f.write("\n\n")
    f.write("=" * 20 + "\n")
    f.write("token entropy distribution")
    write_sorted_table(f, valid_ratio_distribution, stat_dir, "valid_token_ratio")
print(f"stats in {stats_path}")
