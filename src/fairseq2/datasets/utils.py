# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from logging import Logger
from typing import List, TypeVar

import torch

from fairseq2.gang import Gang
from fairseq2.models.seq2seq import Seq2SeqBatch

BatchT = TypeVar("BatchT")


def _all_eod(eod: bool, gang: Gang, logger: Logger) -> bool:
    """Return ``True`` if all processes in ``gang`` have reached end of data."""
    if gang.size == 1:
        return eod

    eods = torch.empty((gang.size,), device=gang.device, dtype=torch.bool)

    gang.all_gather(eods, torch.tensor(eod, device=gang.device))

    if eods.any():
        if logger.isEnabledFor(logging.DEBUG) and not eods.all():
            ranks = ", ".join(str(r) for r in eods.nonzero().squeeze(1).tolist())

            logger.debug(f"End of data reached at rank(s) {ranks}.")

        return True

    return False


def _total_batch_size(batches: List[Seq2SeqBatch], gang: Gang) -> int:
    """Return the sum of the batch sizes of all elements from all ranks."""

    total_bsz = 0
    if gang.size == 1:
        for b in batches:
            total_bsz += b.source_seqs.size(0)
        return total_bsz

    for b in batches:
        batch_sizes = torch.empty((gang.size,), device=gang.device, dtype=torch.int64)
        gang.all_gather(
            batch_sizes, torch.tensor(b.source_seqs.size(0), device=gang.device)
        )
        total_bsz += int(batch_sizes.sum().item())

    return total_bsz
