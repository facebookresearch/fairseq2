# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from fairseq2.data_type import DataType
from fairseq2.device import Device


class Gemma4AudioRMSNorm(Module):
    """RMSNorm matching HuggingFace Gemma4's exact float32 computation.

    Key difference from fairseq2's default ``RMSNorm``:

    - Always computes normalization in float32 (HF casts inputs AND weights
      to float32 before any computation).
    - Casts back to the input dtype at the very end.
    - Does NOT use ``torch.nn.functional.rms_norm`` (which operates in the
      input dtype on PyTorch >= 2.4, producing different results from HF
      when the input is bfloat16).

    This ensures numerical parity with HF's ``Gemma4RMSNorm`` which does::

        normed = self._norm(hidden_states.float())    # float32
        normed = normed * self.weight.float()          # float32
        return normed.type_as(hidden_states)           # cast back
    """

    def __init__(
        self,
        normalized_shape: int,
        bias: bool = False,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            weight = Parameter(
                torch.ones(normalized_shape, device=device, dtype=dtype)
            )
        else:
            weight = None

        self.weight: Parameter | None
        self.register_parameter("weight", weight)

        # Gemma4 audio norms never use bias.
        self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True) + self.eps
        x = x * torch.pow(variance, -0.5)
        if self.weight is not None:
            x = x * self.weight.float()
        return x.to(input_dtype)
