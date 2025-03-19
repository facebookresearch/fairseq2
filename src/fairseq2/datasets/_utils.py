# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import torch

from fairseq2.datasets._error import DatasetLoadError
from fairseq2.gang import Gang, all_sum
from fairseq2.logging import log


def _min_num_batches(num_batches: int, gang: Gang) -> int:
    all_num_batches = torch.zeros((gang.size,), device=gang.device, dtype=torch.int64)

    input_ = torch.tensor([num_batches], device=gang.device)

    gang.all_gather(all_num_batches, input_)

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


def _sum_num_batches(num_batches: int, gang: Gang) -> int:
    total_num_batches = all_sum(gang, num_batches)

    return int(total_num_batches)


def _load_files_and_weights(
    dataset_name: str, path: Path
) -> tuple[list[Path], list[float]]:
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
        raise DatasetLoadError(
            dataset_name, f"The '{manifest_file}' manifest file of the '{dataset_name}' dataset cannot be read. See the nested exception for details."  # fmt: skip
        ) from ex

    # If the directory does not contain a MANIFEST file, treat all JSONL
    # files as part of the dataset with equal weight.
    if content is None:
        try:
            files = list(path.glob("**/*.jsonl"))
        except OSError as ex:
            raise DatasetLoadError(
                dataset_name, f"The JSONL files under the '{path}' directory of the '{dataset_name}' dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
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

        def error() -> DatasetLoadError:
            return DatasetLoadError(
                dataset_name, f"Each line in the '{manifest_file}' manifest file of the '{dataset_name}' dataset must represent a path to a JSONL file and a weight, but line {idx} is '{line}' instead."  # fmt: skip
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
            raise DatasetLoadError(
                dataset_name, f"The '{file}' path referred at line {idx} in the '{manifest_file}' manifest file of the '{dataset_name}' dataset does not exist."  # fmt: skip
            )

        files.append(file)

        try:
            weight = float(fields[1].strip())
        except ValueError:
            raise error() from None

        weights.append(weight)

    return files, weights
