# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase as TestCaseBase

import torch
from torch import Tensor

# Do not print stack frames from this module in assertion failures.
__unittest = True


class TestCase(TestCaseBase):
    # Note that the test runner can change the default device based on the
    # provided command line arguments.
    _device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Specifies the default device that tests should use."""
        return TestCase._device

    def assertAllClose(self, a: Tensor, b: Tensor) -> None:
        """Asserts if ``a`` and ``b`` are element-wise equal within a tolerance."""
        torch.testing.assert_close(a, b)
