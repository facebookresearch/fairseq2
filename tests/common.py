# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from unittest import TestCase as TestCaseBase

import torch
from torch import Tensor

# Do not print stack frames from this module in assertion failures.
__unittest = True


class TestCase(TestCaseBase):
    def assertAllClose(
        self,
        a: Tensor,
        b: Tensor,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        equal_nan: bool = False,
        msg: str = None,
    ) -> None:
        torch.testing.assert_close(a, b)
