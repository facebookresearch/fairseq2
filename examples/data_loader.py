import os
import time
from typing import Any, Dict

import torch
from torch import Tensor

from fairseq2.data import DataPipeline, StringLike, list_files
from fairseq2.data.text import SentencePieceEncoder, SentencePieceModel, read_text
from fairseq2.typing import Device

torch.cuda.init()

# Using Guillaume's IWSLT EN-DE dataset.
dataset_dir = "/checkpoint/guw/fairseq/iwslt14.tokenized.de-en/"

# This is our SentencePiece model API implemented in C++.
spm = SentencePieceModel(
    pathname=os.path.join(dataset_dir, "spm.model"),
    # The `control_tokens` parameter is used to natively add custom tokens (e.g.
    # pad, language markers) directly at the Protobuf level. Here we add <pad>
    # since the IWSLT SentencePiece model does not have a pad token, plus <en>
    # and <de> as beginning-of-sentece markers for a translation task.
    control_tokens=["<pad>", "<en>", "<de>"],
)


# We construct a sub-data pipeline per language (e.g. en, de, fr).
def build_lang_pipeline(
    split: str,
    lang: str,
    encoder: SentencePieceEncoder,
) -> DataPipeline:
    # Retrieve the token index that the SentencePiece model has assigned to the
    # specified language. We use this index to replace the <bos> token.
    lang_token_idx = spm.token_to_index(f"<{lang}>")

    def replace_bos(batch: Tensor) -> Tensor:
        # Replace <bos> with the language token (e.g. <en>, <de>).
        batch[:, 0] = lang_token_idx

        return batch

    def read_tokenized_data(pathname: StringLike) -> DataPipeline:
        return (
            # Open the file at `pathname`, map it to memory, and read it line by
            # line. We trim each line's end, effectively removing the newline
            # character.
            read_text(pathname, rtrim=True, memory_map=True)
            # We batch every 128 lines. We return a partial batch at the end.
            # This is fine since our SentencePiece encoder (see below) handles
            # both vertical and horizontal padding.
            .bucket(128, drop_remainder=False)
            # Tokenize the batch of text lines.
            .map(encoder)
            # Replace <bos> with the language token.
            .map(replace_bos)
            # And construct the pipeline.
            .and_return()
        )

    return (
        # This call is mostly for demonstration purposes since our dataset
        # contains only a single file per split (i.e. train, valid, test). We
        # use it to show that you can in fact open more than one file per split.
        list_files(os.path.join(dataset_dir, "text/raw", f"{split}.{lang}"))
        # For each file, "yield from" the `read_tokenized_data` function.
        .yield_from(read_tokenized_data)
        # And construct the pipeline.
        .and_return()
    )


# This function constructs the complete data pipeline for a language pair.
def build_data_pipeline(
    src_lang: str,
    tgt_lang: str,
    split: str,
) -> DataPipeline:
    # For demonstration purposes, we allocate our batches on the first CUDA
    # device.
    device = Device("cuda:0")

    # Unlike the official SentencePiece API we refactored our encoding/decoding
    # API from the actual model API.
    encoder = SentencePieceEncoder(
        # Use the SentencePiece model represented by the `spm` global variable.
        spm,
        # Enable sampling (a.k.a. regulazation).
        enable_sampling=True,
        # These are the default values for `nbest_size` and `alpha`, we specify
        # them here for demonstration purposes.
        nbest_size=1,
        alpha=0.1,
        # If the number of input text lines is less than 16, pad the batch
        # vertically.
        batch_size=16,
        # We support int16, int32, and int64 (int8 coming later).
        dtype=torch.int16,
        # We are using the lowest-level SentencePiece API which makes it
        # possible for us to tokenize the text directly into the tensor storage.
        device=device,
        # We pin the host memory to speed up the GPU data transfer after
        # tokenization.
        pin_memory=True,
        disable_parallelism=False,
    )

    # Build the sub-data pipelines for the source and target (i.e. en, de)
    # languages.
    src_dp = build_lang_pipeline(split, src_lang, encoder)
    tgt_dp = build_lang_pipeline(split, tgt_lang, encoder)

    # And finally zip both language pipelines into one.
    return DataPipeline.zip([src_dp, tgt_dp]).prefetch(10).and_return()


dp = build_data_pipeline(src_lang="en", tgt_lang="de", split="train")

start_time = time.perf_counter()

num_batches = 0

state: Dict[str, Any] = {}

# Just a noop iterator. Each `batch` is a pair of tensors representing the
# source and target language data.
for batch in dp:
    # Preserve the position of the data pipeline after the 100th iteration.
    if num_batches == 100:
        state = dp.state_dict()

    # Restore the previously saved state; effectively roll back to the 100th
    # iteration and repeat the last 200 batches.
    if num_batches == 300:
        dp.load_state_dict(state)

    num_batches += 1

torch.cuda.synchronize()

elapsed_time = time.perf_counter() - start_time

print(f"We read {num_batches} batches in {elapsed_time:.1f} second(s).")
