###################################
# this file splits input text into prefix and completion by end of sentence to prepare for RL process.
# current it support randomly splitting.
###################################
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq
from tqdm import tqdm

# TODO(lidli): can consider consolidate w/ form_sentence_ppls.py.
INPUT_DIR = "/fsx-chinchilla/xianl/dclm/datasets/finemath_4"
INPUT_FILE_REGEX = r"train-\d+-of-\d+.parquet"
END_OF_SENTENCE_TOKENS = {
    ".",
    "!",
    "?",
}  # TODO: including new-line and space characters
OUTPUT_DIR = "/fsx-ram/lidli/datasets/finemath_4"
MIN_PREFIX_TOKENS: int = 30  # TODO(lidli): can do some analysis.
MIN_COMPLETION_TOKENS: int = 100


# constants
COMPLETION_FIELD: str = "completion"
PREFIX_FIELD: str = "prefix"
THINK_TOKEN: str = "<think>"

# debugging option
DOC_LIMIT: Optional[int] = None


def is_sentence_ending_token(token):
    return token.strip() in END_OF_SENTENCE_TOKENS


def construct_prefix_completion(text):
    # TODO(lidli): currently we do one random split on the condition of min
    # prefix and completion tokens. we can consider other options.
    split_i_list = [
        i for i, char in enumerate(text) if (is_sentence_ending_token(char))
    ]
    if len(split_i_list) == 0:
        return None  # no valid split
    rand_split_i: int = random.choice(split_i_list)
    prefix_text = text[: rand_split_i + 1]
    compl_text = text[rand_split_i + 1 :]
    output: Dict[str, str] = {
        PREFIX_FIELD: f"{prefix_text} {THINK_TOKEN}",
        COMPLETION_FIELD: compl_text,
    }
    return output


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

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"{OUTPUT_DIR=}")

for in_file in tqdm(in_files, desc="files"):
    n_skipped_docs: int = 0
    n_total_docs: int = 0
    print(f"Processing {in_file}")
    in_path = os.path.join(INPUT_DIR, in_file)
    parquet_file = pq.ParquetFile(in_path)
    table = parquet_file.read()
    batches = table.to_batches()
    if DOC_LIMIT is not None:
        print(f"=========DEBUGGING WITH {DOC_LIMIT=}=========")
        print(f"{DOC_LIMIT=}!")
        batches = batches[: min(DOC_LIMIT, len(batches))]

    file_core_outputs, file_metadata_outputs = [], []
    # Iterate over rows
    for batch in tqdm(batches, desc="doc_batches"):
        for datum in batch.to_pylist():
            doc_tokens, doc_logprobs = [], []
            doc_sentences = []
            processed = construct_prefix_completion(datum["text"])

            n_total_docs += 1
            if processed is None:
                n_skipped_docs += 1
                continue

            file_core_outputs.append(processed)

    # print summary
    print(
        f"summary: {in_file=}, {n_skipped_docs=}, {n_total_docs=}, {len(file_core_outputs)=}"
    )
    write_data(OUTPUT_DIR, in_file.replace(".parquet", ".jsonl"), file_core_outputs)
