# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from fairseq2.gang import Gang
from fairseq2.utils.logging import LogWriter


def _reduce_batch_size(batch_size: int, gang: Gang, log: LogWriter) -> int:
    if gang.size == 1:
        return batch_size

    batch_sizes = torch.zeros((gang.size,), device=gang.device, dtype=torch.int64)

    gang.all_gather(batch_sizes, torch.tensor(batch_size, device=gang.device))

    # Check if any process has reached end of data. If so, return 0 to indicate
    # that we should stop the iterator.
    if (eods := batch_sizes == 0).any():
        if log.is_enabled_for(logging.DEBUG) and not eods.all():
            ranks = ", ".join(str(r) for r in eods.nonzero().squeeze(1).tolist())

            log.debug("End of data reached at rank(s) {}.", ranks)

        return 0

    return int(batch_sizes.sum())
