# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Union

import torch
from torch import Tensor

# The default device that tests should use.
#
# Note that pytest can change the default device based on the provided command
# line arguments.
device = torch.device("cpu")


def assert_close(a: Tensor, b: Union[Tensor, List[Any]]) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal within a tolerance."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b)  # type: ignore[attr-defined]


def assert_equal(a: Tensor, b: Union[Tensor, List[Any]]) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b, rtol=0, atol=0)  # type: ignore[attr-defined]


def has_no_inf(a: Tensor) -> bool:
    """Indicate whether ``a`` has no positive or negative infinite element."""
    return not torch.any(torch.isinf(a))


def has_no_nan(a: Tensor) -> bool:
    """Indicate whether  ``a`` has no NaN element."""
    return not torch.any(torch.isnan(a))


def assert_equal_tensor_list(
    actual: Union[Tensor, List[Any]], expected: Union[Tensor, List[Any]]
) -> None:
    """Assert equality of embeded list of tensors"""
    assert type(actual) == type(expected)

    if isinstance(actual, Tensor):
        assert_equal(actual, expected)
    elif isinstance(actual, List):
        for i in range(len(actual)):
            assert_equal_tensor_list(actual[i], expected[i])
    else:
        raise ValueError(f"{type(actual)} not supported.")
