# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.nn import Projection


@final
class SoftcappedProjection(Projection):
    """Projection with tanh softcapping applied to outputs.

    Wraps an existing projection layer and applies tanh softcapping to
    the output logits, as used in Gemma3n for final logit transformation.
    """

    proj: Projection
    softcap: float

    def __init__(self, proj: Projection, softcap: float) -> None:
        """
        :param proj: The base projection layer.
        :param softcap: The softcapping factor. Logits are divided by this
            value, passed through tanh, then multiplied by this value.
        """
        super().__init__(proj.input_dim, proj.output_dim)

        self.proj = proj
        self.softcap = softcap

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply projection and tanh softcapping.

        :param x: Input tensor.
        :returns: Softcapped projection output.
        """
        logits = self.proj(x)
        logits = logits / self.softcap
        logits = torch.tanh(logits)
        logits = logits * self.softcap
        return logits
