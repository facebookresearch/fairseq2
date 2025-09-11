###################################
# this file splits input tokens into prefix and completion based on entropy so
# that our model can learn to reason on more difficult tokens.
###################################

import json
import logging
import os
import random
import re
from collections import Counter
from typing import Any, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


random.seed(5)

ENTROPY_THRESHOLD: float = (
    2  # TODO(lidli): consider upsampling hard tokens rather than doing a filtering.
)
INPUT_DIR = "/fsx-ram/lidli/datasets/natural_reasoning_data_extracted_w_token_entropy"
INPUT_FILE_REGEX = r"data.chunk.\d+.jsonl"
OUPUT_DIR = f"/fsx-ram/lidli/datasets/natural_reasoning_data_extracted_perplexity_scores_1B_split_for_rl_entropy_{str(ENTROPY_THRESHOLD).replace('.','_')}_min_100_compl_handle_whitespace"
MIN_PREFIX_TOKENS: int = 30  # TODO(lidli): can do some analysis.
MIN_COMPLETION_TOKENS: int = 100
WHITE_SPACE_SET = {"\n", " ", "\t", "\r"}

# constants
COMPLETION_FIELD: str = "completion"
PREFIX_END_FIELD: str = "prefix_token_end"
REASON_END_WRAP_FIELD: str = "reason_end_wrap"
REASON_START_WRAP_FIELD: str = "reason_start_wrap"
PREFIX_FIELD: str = "prefix"
TOKENS_FIELD: str = "tokens"
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
train_dir = os.path.join(OUPUT_DIR, "train")
metadata_dir = os.path.join(OUPUT_DIR, "metadata")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)


print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, use_fast=False)
print("finished loadding tokenizer")

eligible_split_ratios = []
entropy_counter = Counter()
for in_file in tqdm(in_files, desc="files"):
    n_skipped_docs: int = 0
    n_total_docs: int = 0
    in_path = os.path.join(INPUT_DIR, in_file)
    print(f"Processing {in_path}")
    with open(in_path, "r", encoding="utf-8") as in_f:
        docs = in_f.readlines()

    outputs = []
    metadata = []

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

        if len(prompt_token_ids) < MIN_COMPLETION_TOKENS + MIN_PREFIX_TOKENS:
            n_skipped_docs += 1
            continue
        if prompt_entropies[0] is None:
            prompt_entropies = prompt_entropies[1:]
        entropy_counter.update(round(entropy, 1) for entropy in prompt_entropies)
        assert len(prompt_token_ids) == len(prompt_entropies) + 1
        token_i_list = []  # stores completion start tok indexes

        # logger.debug(f"=========={doc_data['text']=}")
        # logger.debug("==========valid completion start")
        # logger.debug(
        #     tokenizer.decode(
        #         prompt_token_ids[
        #             MIN_PREFIX_TOKENS : len(prompt_token_ids)
        #             - MIN_COMPLETION_TOKENS
        #             + 1
        #         ],
        #         add_special_tokens=False,
        #     )
        # )

        for i in range(
            MIN_PREFIX_TOKENS, len(prompt_token_ids) - MIN_COMPLETION_TOKENS + 1
        ):
            compl_start_tok = tokenizer.decode(
                prompt_token_ids[i], add_special_tokens=False
            )
            prefix_end_tok = tokenizer.decode(
                prompt_token_ids[i - 1], add_special_tokens=False
            )
            # logger.debug(f"{prefix_end_tok}|{compl_start_tok}")
            if (
                # completion tokens should not start with purely whitespace tokens
                (compl_start_tok not in WHITE_SPACE_SET)
                # completion starts with ws or prefix ends with ws. don't split in the mid of words.
                and any(
                    prefix_end_tok.endswith(ws) or compl_start_tok.startswith(ws)
                    for ws in WHITE_SPACE_SET
                )
            ):
                token_i_list.append(i)
                # logger.debug(f"{i=}")

        token_i_list_high_entropy = [
            i for i in token_i_list if prompt_entropies[i - 1] >= ENTROPY_THRESHOLD
        ]  # NOTE: entropy index is tok index - 1

        logger.debug(f"{token_i_list=}")
        logger.debug(f"{token_i_list_high_entropy=}")
        logger.debug(f"{prompt_entropies=}")

        curr_eligible_split_ratio = len(token_i_list_high_entropy) / len(
            prompt_token_ids
        )
        eligible_split_ratios.append(curr_eligible_split_ratio)

        if token_i_list_high_entropy:
            tok_split_i = random.choice(token_i_list_high_entropy)
        elif token_i_list:
            tok_split_i = random.choice(token_i_list)
        else:
            logger.warning(
                f"skipping! no valid split at whitespace!\n{doc_data['text']}"
            )
            n_skipped_docs += 1
            continue
        output = {}
        output[PREFIX_FIELD] = tokenizer.decode(
            prompt_token_ids[:tok_split_i],
            add_special_tokens=False,
        ).removeprefix(BOS_TOKEN)
        output[COMPLETION_FIELD] = tokenizer.decode(
            prompt_token_ids[tok_split_i:], add_special_tokens=False
        )
        output["org_tokens_at_split"] = prompt_token_ids[
            tok_split_i - 1 : tok_split_i + 1
        ]  # two tokens: prefix end token + completion start token
        output["org_text_at_split"] = (
            output[PREFIX_FIELD][-10:] + "|" + output[COMPLETION_FIELD][:10]
        )  # two bytes: prefix end + completion start
        # original whitespace
        try:
            ws_match = next(
                ws
                for ws in WHITE_SPACE_SET
                if tokenizer.decode(
                    prompt_token_ids[tok_split_i - 1], add_special_tokens=False
                ).endswith(ws)
                or tokenizer.decode(
                    prompt_token_ids[tok_split_i], add_special_tokens=False
                ).startswith(ws)
            )
        except StopIteration:
            raise Exception(f"{prompt_token_ids=}\n{tok_split_i=}")
        # mirrow the whitespace to wrap think tags when needed
        output[REASON_START_WRAP_FIELD] = ""
        if not any(output[PREFIX_FIELD].endswith(ws) for ws in WHITE_SPACE_SET):
            output[REASON_START_WRAP_FIELD] += ws_match
        output[REASON_START_WRAP_FIELD] += "<think>"
        output[PREFIX_FIELD] += output[REASON_START_WRAP_FIELD]

        output[REASON_END_WRAP_FIELD] = "</think>"
        if not any(output[COMPLETION_FIELD].startswith(ws) for ws in WHITE_SPACE_SET):
            output[REASON_END_WRAP_FIELD] += ws_match

        output["merged_text_at_split"] = (
            output[PREFIX_FIELD][-30:]
            + "<reason>"
            + output[REASON_END_WRAP_FIELD]
            + output[COMPLETION_FIELD][:30]
        )  # what the final text would be around the split

        # output[TOKENS_FIELD] = prompt_token_ids
        # output[PREFIX_END_FIELD] = tok_split_i
        n_total_docs += 1
        outputs.append(
            {
                key: output[key]
                for key in [PREFIX_FIELD, COMPLETION_FIELD, REASON_END_WRAP_FIELD]
            }
        )
        metadata.append(output)

    # print summary
    print(f"summary: {in_file=}, {n_skipped_docs=}, {n_total_docs=}, {len(outputs)=}")
    write_data(train_dir, in_file, outputs)
    write_data(metadata_dir, in_file, metadata)

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
