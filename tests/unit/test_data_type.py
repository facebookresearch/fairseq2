# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from fairseq2.data_type import DataType, current_dtype, get_current_dtype
from tests.common import device


def test_current_dtype_works() -> None:
    def check_dtype(dtype: DataType) -> None:
        t = torch.ones((4, 4), device=device)

        assert t.dtype == dtype

        assert get_current_dtype() == dtype

    with current_dtype(torch.bfloat16):
        check_dtype(torch.bfloat16)

        with current_dtype(torch.float16):
            check_dtype(torch.float16)

        check_dtype(torch.bfloat16)

    check_dtype(torch.float32)
