# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sound projection MLP for the NemotronH audio encoder.

Projects Parakeet audio encoder outputs (hidden_size=1024) into the
language model's hidden space (model_dim=2688):

    RMSNorm(1024) → Linear(1024 → 4096) → SquaredReLU → Linear(4096 → 2688)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.nn import Linear, RMSNorm


@final
class SoundProjection(Module):
    """Projects audio encoder output to language model hidden space.

    Architecture:
        ``RMSNorm`` → ``Linear`` → ``SquaredReLU`` → ``Linear``

    The SquaredReLU activation is ``ReLU(x)^2``, matching the HF implementation.
    """

    def __init__(
        self,
        encoder_dim: int,
        model_dim: int,
        hidden_dim: int,
        *,
        bias: bool = False,
        eps: float = 1e-5,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoder_dim:
            The audio encoder output dimension (e.g. 1024).
        :param model_dim:
            The language model hidden dimension (e.g. 2688).
        :param hidden_dim:
            The intermediate hidden dimension (e.g. 4096).
        :param bias:
            If ``True``, the linear layers use bias.
        :param eps:
            The epsilon for RMSNorm.
        """
        super().__init__()

        self.norm = RMSNorm(encoder_dim, bias=False, eps=eps, device=device, dtype=dtype)

        self.linear1 = Linear(
            encoder_dim, hidden_dim, bias=bias, device=device, dtype=dtype
        )
        self.linear2 = Linear(
            hidden_dim, model_dim, bias=bias, device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project audio features to language model space.

        :param x:
            Audio encoder output. *Shape:* ``[B, T, encoder_dim]``.

        :returns:
            Projected features. *Shape:* ``[B, T, model_dim]``.
        """
        x = self.norm(x)
        x = self.linear1(x)
        # SquaredReLU: ReLU(x)^2
        x = torch.relu(x).square()
        x = self.linear2(x)
        return x

    if TYPE_CHECKING:
        __call__ = forward

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"encoder_dim={self.norm.normalized_shape}, "
            f"model_dim={self.linear2.output_dim}"
        )
