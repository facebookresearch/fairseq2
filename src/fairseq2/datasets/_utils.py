# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fairseq2.datasets.error import DatasetError
from fairseq2.error import InfraError


def _load_files_and_weights(
    dataset_name: str, path: Path
) -> tuple[list[Path], list[float]]:
    path = path.expanduser().resolve()

    if not path.is_dir():
        return [path], [1.0]

    manifest_file = path.joinpath("MANIFEST")

    try:
        with manifest_file.open(encoding="utf-8") as fp:
            content = list(fp)
    except FileNotFoundError:
        content = None
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while reading the '{manifest_file}' manifest file of the '{dataset_name}' dataset. See the nested exception for details."  # fmt: skip
        ) from ex

    # If the directory does not contain a MANIFEST file, treat all JSONL
    # files as part of the dataset with equal weight.
    if content is None:
        try:
            files = list(path.glob("**/*.jsonl"))
        except OSError as ex:
            raise InfraError(
                f"A system error has occurred while retrieving the list of JSONL files under the '{path}' directory of the '{dataset_name}' dataset. See the nested exception for details."  # fmt: skip
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

        def error() -> DatasetError:
            return DatasetError(
                dataset_name, f"Each line in the '{manifest_file}' manifest file of the '{dataset_name}' dataset is expected to represent a path to a JSONL file and a weight, but line {idx} is '{line}' instead."  # fmt: skip
            )

        fields = line.rstrip().split("\t")

        if len(fields) != 2:
            raise error()

        file_path = fields[0].strip()
        if not file_path:
            raise error()

        try:
            file = path.joinpath(file_path)
        except ValueError:
            raise error() from None

        if not file.exists():
            raise DatasetError(
                dataset_name, f"The '{file}' path referred at line {idx} in the '{manifest_file}' manifest file does not exist."  # fmt: skip
            )

        files.append(file)

        try:
            weight = float(fields[1].strip())
        except ValueError:
            raise error() from None

        weights.append(weight)

    return files, weights


class DynamicBatcher:
    def __init__(self, max_sentences: int, max_tokens: int, bsz_mult: int) -> None:
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        self.bsz_mult = bsz_mult

        self.tail_overflow = False
        self.overflow = False

        self.max_batch_tokens = 0
        self.max_tail_tokens = 0
        self.num_sentences = 0
        self.batch_sentences = 0

    def cost_fn(self, example: dict[str, Any]) -> float:
        audio_size = example["audio_size"]

        if audio_size == -1:
            return 1

        self.max_tail_tokens = max(self.max_tail_tokens, audio_size)
        self.num_sentences += 1
        self.overflow = (
            self.num_sentences > self.max_sentences > 0
            or self.num_sentences * max(self.max_batch_tokens, self.max_tail_tokens)
            > self.max_tokens
            > 0
        )
        size_matches_with_bsz_mult = (
            self.num_sentences < self.bsz_mult
            or self.num_sentences % self.bsz_mult == 0
        )

        if self.overflow:
            self.tail_overflow = (
                (self.max_tail_tokens * (self.num_sentences - self.batch_sentences))
                > self.max_tokens
                > 0
            )
            self.max_batch_tokens = self.max_tail_tokens
            return 1

        if size_matches_with_bsz_mult:
            self.batch_sentences = self.num_sentences
            self.max_batch_tokens = max(self.max_batch_tokens, self.max_tail_tokens)
            self.max_tail_tokens = 0
        return 0

    def bucket_creation_fn(
        self, bucket: Sequence[Any]
    ) -> tuple[Sequence[Sequence[Any]], Sequence[Any]]:
        ret = ([bucket[: self.batch_sentences]], bucket[self.batch_sentences :])

        if self.tail_overflow:
            self.tail_overflow = False
            self.overflow = False

            ret = (
                [bucket[: self.batch_sentences], bucket[self.batch_sentences : -1]],
                [bucket[-1]],
            )

            self.max_batch_tokens = bucket[-1]["audio_size"]
            self.batch_sentences = 1
            self.num_sentences = 1

        elif self.overflow:
            self.overflow = False

            self.max_batch_tokens = self.max_tail_tokens
            self.batch_sentences = self.num_sentences - self.batch_sentences
            self.num_sentences = self.batch_sentences

        self.max_tail_tokens = 0
        return ret
