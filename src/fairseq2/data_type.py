# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeAlias

import torch

DataType: TypeAlias = torch.dtype


@contextmanager
def default_dtype(dtype: DataType) -> Iterator[None]:
    original_dtype = torch.get_default_dtype()

    torch.set_default_dtype(dtype)

    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)
