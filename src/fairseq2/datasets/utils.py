# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from logging import Logger

import torch

from fairseq2.gang import Gang


def all_eod(eod: bool, gang: Gang, logger: Logger) -> bool:
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
