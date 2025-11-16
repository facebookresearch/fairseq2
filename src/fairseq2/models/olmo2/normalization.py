# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO2-specific normalization layers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Size, Tensor
from torch.nn import Parameter
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import RMSNorm as BaseRMSNorm


class OLMO2RMSNorm(BaseRMSNorm):
    """OLMO2-specific RMS Normalization.

    The mathmatical represent of this RMSNorm is identical to the LLAMA implementation.

    The key difference from standard RMSNorm is the order of operations:
    - Standard: normalize -> cast to original dtype -> multiply by weight
    - OLMO2:    normalize -> multiply by weight     -> cast to original dtype

    This matches the [HuggingFace OLMO2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modular_olmo2.py) where the weight
    and normalized hidden states are multiplied before converting back
    to the input dtype.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype

        # For numerical stability, normalize in single precision
        x = x.float()

        # Normalize
        x = self._normalize(x)

        # Apply weight BEFORE casting back (OLMO2-specific)
        if self.weight is not None:
            x = x * self.weight

            if self.bias is not None:
                x = x + self.bias

        # Cast back to original dtype
        return x.to(input_dtype)
