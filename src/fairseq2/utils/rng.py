# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch


def seed(value: int) -> None:
    """Set RNG seed for ``random``, ``np.random``, and ``torch``.

    :param value:
        The new seed.
    """
    if value >= 1 << 32:
        raise ValueError(
            f"`value` must be greater than or equal to 0 and less than 2^32, but is {value} instead."
        )

    random.seed(value)

    np.random.seed(value)

    torch.manual_seed(value)


def use_deterministic(value: bool, warn_only: bool = False) -> None:
    """Set whether PyTorch algorithms must use deterministic algorithms.

    :param value:
        If ``True``, uses deterministic algorithms.
    :param warn_only:
        If ``True``, operations that do not have a deterministic implementation
        will raise a warning instead of an error.
    """
    torch.backends.cudnn.benchmark = not value

    torch.use_deterministic_algorithms(value, warn_only=warn_only)
