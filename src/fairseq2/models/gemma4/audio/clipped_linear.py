# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

from fairseq2.data_type import DataType
from fairseq2.device import Device


class Gemma4ClippedLinear(Module):
    """Linear layer with optional input/output clamping.

    Matches HuggingFace's ``Gemma4ClippableLinear`` which wraps ``nn.Linear``
    with four clipping buffers (``input_min``, ``input_max``, ``output_min``,
    ``output_max``).  When ``use_clipping`` is ``True`` the forward pass
    clamps the input and output:

        x = clamp(x, input_min, input_max)
        x = F.linear(x, weight, bias)
        x = clamp(x, output_min, output_max)

    The clipping bounds are loaded from the checkpoint as non-persistent
    buffers (they are **not** trainable parameters).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        use_clipping: bool = True,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_clipping = use_clipping

        self.weight = Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        if bias:
            self.bias: Parameter | None = Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.bias = None

        # Clipping buffers — initialised to ±inf (no-op) and overwritten by
        # the checkpoint via ``load_state_dict``.
        if use_clipping:
            self.register_buffer(
                "input_min",
                torch.tensor(-float("inf"), device=device),
                persistent=True,
            )
            self.register_buffer(
                "input_max",
                torch.tensor(float("inf"), device=device),
                persistent=True,
            )
            self.register_buffer(
                "output_min",
                torch.tensor(-float("inf"), device=device),
                persistent=True,
            )
            self.register_buffer(
                "output_max",
                torch.tensor(float("inf"), device=device),
                persistent=True,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_clipping:
            x = torch.clamp(x, self.input_min, self.input_max)  # type: ignore[arg-type]

        x = F.linear(x, self.weight, self.bias)

        if self.use_clipping:
            x = torch.clamp(x, self.output_min, self.output_max)  # type: ignore[arg-type]

        return x
