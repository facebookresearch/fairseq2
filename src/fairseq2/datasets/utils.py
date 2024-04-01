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


def _total_batch_size_and_eod(
    batches: List[BatchT], eod: bool, gang: Gang, logger: Logger
) -> Tuple[int, bool]:
    """
    Return the sum of the batch sizes of all elements from all ranks and
    return ``True`` if all processes in ``gang`` have reached end of data.
    """
    assert gang.size > 1

    batch_sizes_and_eods = torch.zeros(
        (gang.size, 2), device=gang.device, dtype=torch.int64
    )
    gang.all_gather(
        batch_sizes_and_eods,
        torch.tensor(
            [sum(batch.batch_size for batch in batches), int(eod)], device=gang.device
        ),
    )

    batch_sizes, eods = batch_sizes_and_eods.split(1, dim=1)
    batch_sizes = batch_sizes.squeeze(1)
    eods = eods.squeeze(1)

    total_batch_size = int(batch_sizes.sum().item())

    if eods.any():
        if logger.isEnabledFor(logging.DEBUG) and not eods.all():
            ranks = ", ".join(str(r) for r in eods.nonzero().squeeze(1).tolist())
            logger.debug(f"End of data reached at rank(s) {ranks}.")

        return total_batch_size, True

    return total_batch_size, False
