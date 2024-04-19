# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Tuple

import torch

from fairseq2.gang import Gang
from fairseq2.utils.logging import LogWriter


def _reduce_batch_stats(
    stats: List[Tuple[int, int]], gang: Gang, log: LogWriter
) -> Tuple[int, int, int]:
    # (G, N, 2)
    all_stats = torch.zeros(
        (gang.size, len(stats), 2), device=gang.device, dtype=torch.int64
    )

    # (N, 2)
    this_stats = torch.tensor(stats, device=gang.device)

    gang.all_gather(all_stats, this_stats)

    # (G, N)
    batch_sizes = all_stats[:, :, 0::2].squeeze(-1)

    # (G, N)
    num_target_elements = all_stats[:, :, 1::2].squeeze(-1)

    # Determine the number of batches read by each process.
    # (G)
    num_batches = batch_sizes.count_nonzero(dim=-1)

    min_num_batches = int(num_batches.min())
    if min_num_batches == 0:
        # If not all processes reached end of data, report the ones that have
        # reached for debugging purposes.
        if log.is_enabled_for(logging.DEBUG) and num_batches.sum() > 0:
            ranks = num_batches.bool().logical_not_().nonzero().squeeze(-1).tolist()

            s = ", ".join(str(r) for r in ranks)

            log.debug("End of data reached at rank(s) {}.", s)

        return 0, 0, 0

    batch_sizes = batch_sizes[:, :min_num_batches]

    num_target_elements = num_target_elements[:, :min_num_batches]

    return min_num_batches, int(batch_sizes.sum()), int(num_target_elements.sum())
