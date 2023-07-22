# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairseq2.nn.utils.grad import scale_grad
from tests.common import assert_close, device


def test_scale_grad_scales_gradient_correctly() -> None:
    a = torch.full((10, 10), 2.0, device=device, requires_grad=True)

    b = scale_grad(a, 0.1)

    c = b**3.0

    g = torch.autograd.grad(c, a, grad_outputs=torch.ones_like(b))

    expected_grad = torch.full((10, 10), 1.2, device=device)

    assert_close(g[0], expected_grad)


def test_scale_grad_raises_error_if_tensor_is_non_float() -> None:
    a = torch.ones((2, 2), dtype=torch.int32)

    with pytest.raises(
        TypeError,
        match=r"^`x` must be a float tensor, but is of type `torch\.int32` instead\.$",
    ):
        scale_grad(a, 1.0)
