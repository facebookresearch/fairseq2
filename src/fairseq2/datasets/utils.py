# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import numpy as np
import torch

from fairseq2.datasets.error import DatasetError
from fairseq2.gang import Gang
from fairseq2.logging import LogWriter


def _reduce_num_batches(num_batches: int, gang: Gang, log: LogWriter) -> int:
    all_num_batches = torch.zeros((gang.size,), device=gang.device, dtype=torch.int64)

    num_batches_ = torch.tensor(num_batches, device=gang.device)

    gang.all_gather(all_num_batches, num_batches_)

    min_num_batches = int(all_num_batches.min())
    if min_num_batches != 0:
        return min_num_batches

    # If not all processes have reached end of data, report the ones that have
    # reached for debugging purposes.
    if log.is_enabled_for_debug() and all_num_batches.sum() > 0:
        ranks = all_num_batches.bool().logical_not_().nonzero().squeeze(-1).tolist()

        s = ", ".join(str(r) for r in ranks)

        log.debug("End of data reached at rank(s) {}.", s)

    return 0


def _load_files_and_weights(path: Path) -> tuple[list[Path], list[float]]:
    path = path.expanduser().resolve()

    if not path.is_dir():
        return [path], [1.0]

    manifest_file = path.joinpath("MANIFEST")

    try:
        with manifest_file.open() as fp:
            content = list(fp)
    except FileNotFoundError:
        content = None
    except OSError as ex:
        raise RuntimeError(
            f"{manifest_file} cannot be read. See nested exception for details."
        ) from ex

    # If the directory does not contain a MANIFEST file, treat all JSONL
    # files as part of the dataset with equal weight.
    if content is None:
        try:
            files = list(path.glob("**/*.jsonl"))
        except OSError as ex:
            raise RuntimeError(
                f"The JSONL files under {path} cannot be retrieved. See nested exception for details."
            ) from ex

        weights = [1.0 for _ in range(len(files))]

        return files, weights

    # Sort the JSONL files in alphabetical order.
    content.sort()

    files = []

    weights = []

    # Each line of the MANIFEST file corresponds to the path of a JSONL file
    # and its weight (e.g. number of examples).
    for idx, line in enumerate(content):

        def raise_error() -> NoReturn:
            raise DatasetError(
                f"Each line in {manifest_file} must represent a path to a JSONL file and a weight, but line {idx} is '{line}' instead."
            )

        fields = line.rstrip().split("\t")

        if len(fields) != 2:
            raise_error()

        file_path = fields[0].strip()
        if not file_path:
            raise_error()

        try:
            file = path.joinpath(file_path)
        except ValueError:
            raise_error()

        if not file.exists():
            raise DatasetError(
                f"The file '{file}' referred at line {idx} in {manifest_file} does not exist."
            )

        files.append(file)

        try:
            weight = float(fields[1].strip())
        except ValueError:
            raise_error()

        weights.append(weight)

    return files, weights


def batch_by_size_vec(
    indices, num_tokens_vec, max_tokens: int, max_sentences: int, bsz_mult: int
):
    if indices.size == 0:
        return []

    assert (
        max_tokens <= 0 or np.max(num_tokens_vec) <= max_tokens
    ), f"Sentences lengths should not exceed max_tokens={max_tokens}"

    indices_len = len(indices)
    batches_ends = np.zeros(indices_len, dtype=np.int32)
    pos = 0
    new_batch_end = 0
    new_batch_max_tokens = 0
    new_batch_sentences = 0
    new_batch_num_tokens = 0
    overflow = False
    size_matches_with_bsz_mult = False
    batches_count = 0
    batch_start = 0
    tail_max_tokens = 0
    batch_max_tokens = 0

    for pos in range(indices_len):
        tail_max_tokens = max(tail_max_tokens, num_tokens_vec[pos])
        new_batch_end = pos + 1
        new_batch_max_tokens = max(batch_max_tokens, tail_max_tokens)
        new_batch_sentences = new_batch_end - batch_start
        new_batch_num_tokens = new_batch_sentences * new_batch_max_tokens
        overflow = (
            new_batch_sentences > max_sentences > 0
            or new_batch_num_tokens > max_tokens > 0
        )
        size_matches_with_bsz_mult = (
            new_batch_sentences < bsz_mult or new_batch_sentences % bsz_mult == 0
        )
        if overflow:
            tail_num_tokens = tail_max_tokens * (
                new_batch_end - batches_ends[batches_count]
            )
            tail_overflow = tail_num_tokens > max_tokens > 0
            if tail_overflow:
                batches_count += 1
                batches_ends[batches_count] = pos
                tail_max_tokens = num_tokens_vec[pos]
            batch_start = batches_ends[batches_count]
            batches_count += 1
            new_batch_max_tokens = tail_max_tokens
        if overflow or size_matches_with_bsz_mult:
            batches_ends[batches_count] = new_batch_end
            batch_max_tokens = new_batch_max_tokens
            tail_max_tokens = 0

    if batches_ends[batches_count] != indices_len:
        batches_count += 1

    return np.split(indices, batches_ends[:batches_count])


def batch_by_size_fn(
    indices, num_tokens_fn, max_tokens: int, max_sentences: int, bsz_mult: int
):
    indices_len = len(indices)
    num_tokens_vec = np.zeros(indices_len, dtype=np.int64)
    for pos in range(indices_len):
        num_tokens_vec[pos] = num_tokens_fn(indices[pos])
    return batch_by_size_vec(
        indices, num_tokens_vec, max_tokens, max_sentences, bsz_mult
    )
