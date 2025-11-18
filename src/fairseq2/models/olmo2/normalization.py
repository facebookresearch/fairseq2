# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO2-specific normalization layers.

Note: OLMO2RMSNorm inherits from RMSNorm (marked @final) because the only
difference is the order of weight application in forward(). Reimplementing
the entire class would duplicate ~55 lines of boilerplate code for __init__,
reset_parameters, and _normalize. The type checker warning is suppressed as
this is a legitimate architectural need specific to OLMO2's design.
"""

from __future__ import annotations

from torch import Tensor
from typing_extensions import override

from fairseq2.nn import RMSNorm as BaseRMSNorm


class OLMO2RMSNorm(BaseRMSNorm):  # type: ignore[misc]
    """OLMO2-specific RMS Normalization.

    The mathematical representation of this RMSNorm is identical to the LLAMA implementation.

    The key difference from standard RMSNorm is the order of operations:
    - Standard: normalize -> cast to original dtype -> multiply by weight
    - OLMO2:    normalize -> multiply by weight     -> cast to original dtype

    This matches the HuggingFace OLMO2 implementation where the weight
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
