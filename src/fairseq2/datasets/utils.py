# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from logging import Logger
from typing import List, Tuple, TypeVar

import torch

from fairseq2.gang import Gang
from fairseq2.models import Batch

BatchT = TypeVar("BatchT", bound=Batch)


def _all_batch_sizes(
    batches: List[BatchT], gang: Gang, logger: Logger
) -> Tuple[int, bool]:
    """Return the sum of the batch sizes of all elements from all ranks."""

    total_bsz = 0
    if gang.size == 1:
        for batch in batches:
            total_bsz += batch.batch_size
        return total_bsz, total_bsz == 0

    eod = False
    for batch in batches:
        batch_sizes = torch.empty((gang.size,), device=gang.device, dtype=torch.int64)
        gang.all_gather(batch_sizes, torch.tensor(batch.batch_size, device=gang.device))
        if (batch_sizes == 0).any():
            eod = True
            if logger.isEnabledFor(logging.DEBUG) and not (batch_sizes == 0).all():
                ranks = ", ".join(
                    str(r) for r in batch_sizes.nonzero().squeeze(1).tolist()
                )

                logger.debug(f"End of data reached at rank(s) {ranks}.")

        total_bsz += int(batch_sizes.sum().item())

    return total_bsz, eod
