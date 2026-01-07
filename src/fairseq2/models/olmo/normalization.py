# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO-specific normalization layers."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import LayerNorm


class OLMORMSNorm(LayerNorm):
    """OLMO Root Mean Square Layer Normalization.

    The mathematical representation of this RMSNorm is identical to the LLaMA implementation.

    The key difference from standard RMSNorm is the order of operations:
    - Standard: normalize -> cast to original dtype -> multiply by weight
    - OLMO:     normalize -> multiply by weight     -> cast to original dtype

    This matches the [HuggingFace OLMO implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo/modular_olmo.py) where the weight
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

        # Apply weight BEFORE casting back (OLMO-specific)
        if self.weight is not None:
            x = x * self.weight

            if self.bias is not None:
                x = x + self.bias

        # Cast back to original dtype
        return x.to(input_dtype)
