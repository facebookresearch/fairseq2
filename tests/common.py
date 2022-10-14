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
    # The default device that tests should use.
    #
    # Note that the test runner can change the default device based on the
    # provided command line arguments.
    device = torch.device("cpu")

    def assertAllClose(self, a: Tensor, b: Tensor) -> None:
        """Asserts if ``a`` and ``b`` are element-wise equal within a tolerance."""
        torch.testing.assert_close(a, b)
