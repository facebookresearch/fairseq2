# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor

from fairseq2.device import CPU, Device
from fairseq2.typing import ContextManager
from fairseq2.utils.rng import RngBag

# The default device that tests should use. Note that pytest can change it based
# on the provided command line arguments.
device = CPU


def assert_close(
    a: Tensor,
    b: Tensor | float | list[float],
    atol: float = 1.0e-05,
    rtol: float = 1.3e-06,
) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal within a tolerance."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)  # type: ignore[attr-defined]


def assert_equal(a: Tensor, b: Tensor | int | list[int]) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b, atol=0, rtol=0)  # type: ignore[attr-defined]


def has_no_inf(a: Tensor) -> bool:
    """Return ``True`` if ``a`` has no positive or negative infinite element."""
    return not torch.any(torch.isinf(a))


def has_no_nan(a: Tensor) -> bool:
    """Return ``True`` if ``a`` has no NaN element."""
    return not torch.any(torch.isnan(a))


def temporary_manual_seed(seed: int, *devices: Device) -> ContextManager[None]:
    """Temporarily changes the seed of the RNGs of ``devices``."""
    rng_bag = RngBag.from_device_defaults(*devices)

    return rng_bag.temporary_manual_seed(seed)
