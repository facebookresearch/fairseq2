# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import typing as tp
import unittest

import torch
from torch import Tensor

from fairseq2.typing import Device

# Do not print stack frames from this module in assertion failures.
__unittest = True


class TestCase(unittest.TestCase):
    # The default device that tests should use.
    #
    # Note that the test runner can change the default device based on the
    # provided command line arguments.
    device = torch.device("cpu")

    def assertAllClose(self, a: Tensor, b: tp.Union[Tensor, tp.List[tp.Any]]) -> None:
        """Asserts if ``a`` and ``b`` are element-wise equal within a tolerance."""
        assert_close(a, b)


def assert_close(a: Tensor, b: tp.Union[Tensor, tp.List[tp.Any]]) -> None:
    """Asserts if ``a`` and ``b`` are element-wise equal within a tolerance."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b)
    torch.testing.assert_close(a, b)  # type: ignore[attr-defined]


def assert_equal(a: Tensor, b: tp.Union[Tensor, tp.List[tp.Any]]) -> None:
    if not isinstance(b, Tensor):
        b = torch.tensor(b)

    torch.testing.assert_close(a, b, rtol=0, atol=0)  # type: ignore[attr-defined]


def assert_no_inf(a: Tensor) -> None:
    values = a.tolist()
    assert torch.inf not in values
    assert -torch.inf not in values
    assert torch.nan not in values


@contextlib.contextmanager
def tmp_rng_seed(device: Device, seed: int = 0) -> tp.Generator[None, None, None]:
    """Sets a temporary manual RNG seed.

    The RNG is reset to its original state once the block is exited.
    """
    device = torch.device(device)

    if device.type == "cuda":
        devices = [device]
    else:
        devices = []

    with torch.random.fork_rng(devices):
        torch.manual_seed(seed)

        yield
