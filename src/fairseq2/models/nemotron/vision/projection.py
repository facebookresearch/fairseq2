# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vision projection for the NemotronH C-RADIO vision encoder.

Includes pixel_shuffle (spatial downsampling) and the MLP projector (mlp1)
that maps vision features into the language model's hidden space.

Pixel shuffle (v2):
    Merges 2×2 spatial blocks into single tokens with 4× channel expansion:
    [B, h, w, 1280] → [B, h/2, w/2, 5120]

Vision projection (mlp1):
    RMSNorm(5120) → Linear(5120 → 20480) → SquaredReLU → Linear(20480 → 2688)

HF state dict keys:
    mlp1.0.weight  → [5120]      (RMSNorm weight)
    mlp1.1.weight  → [20480, 5120]  (Linear, no bias)
    mlp1.3.weight  → [2688, 20480]  (Linear, no bias)
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


def pixel_shuffle(x: Tensor, grid_h: int, grid_w: int, scale_factor: float = 0.5) -> Tensor:
    """Pixel shuffle (spatial downsampling) for vision features.

    Merges spatial blocks into single tokens with expanded channels.
    For scale_factor=0.5, merges 2×2 blocks: N patches → N/4 tokens, D → 4D.

    This is the "v2" pixel shuffle used in InternVL/NemotronH:
        reshape → permute → flatten channels

    :param x:
        Patch features from ViT. *Shape:* ``[B, N, D]`` where N = grid_h * grid_w.
    :param grid_h:
        Number of patches in the height dimension.
    :param grid_w:
        Number of patches in the width dimension.
    :param scale_factor:
        Spatial downsampling factor (0.5 = merge 2×2 blocks).

    :returns:
        Spatially downsampled features. *Shape:* ``[B, N * scale^2, D / scale^2]``.
        For scale=0.5: ``[B, N/4, 4*D]``.
    """
    b, n, d = x.shape
    assert n == grid_h * grid_w, f"Expected {grid_h * grid_w} patches, got {n}"

    h_new = int(grid_h * scale_factor)
    w_new = int(grid_w * scale_factor)
    merge_h = grid_h // h_new  # 2 for scale=0.5
    merge_w = grid_w // w_new  # 2 for scale=0.5

    # [B, N, D] → [B, grid_h, grid_w, D]
    x = x.reshape(b, grid_h, grid_w, d)

    # [B, grid_h, grid_w, D] → [B, h_new, merge_h, w_new, merge_w, D]
    x = x.reshape(b, h_new, merge_h, w_new, merge_w, d)

    # [B, h_new, merge_h, w_new, merge_w, D] → [B, h_new, w_new, merge_h, merge_w, D]
    x = x.permute(0, 1, 3, 2, 4, 5)

    # Flatten spatial merge and channel dims: [B, h_new, w_new, merge_h * merge_w * D]
    x = x.reshape(b, h_new * w_new, merge_h * merge_w * d)

    return x


@final
class VisionProjection(Module):
    """Projects vision features to language model hidden space.

    Architecture (matching HF mlp1):
        ``RMSNorm`` → ``Linear`` → ``SquaredReLU`` → ``Linear``

    Same architecture as SoundProjection but with different dimensions:
        - encoder_dim: 5120 (post-pixel-shuffle: 4 * 1280)
        - hidden_dim: 20480
        - model_dim: 2688

    HF key mapping:
        mlp1.0 → norm (RMSNorm)
        mlp1.1 → linear1 (Linear, no bias)
        mlp1.3 → linear2 (Linear, no bias)
        (mlp1.2 is SquaredReLU activation, no parameters)
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
            The vision encoder output dimension after pixel shuffle (e.g. 5120).
        :param model_dim:
            The language model hidden dimension (e.g. 2688).
        :param hidden_dim:
            The intermediate hidden dimension (e.g. 20480).
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
        """Project vision features to language model space.

        :param x:
            Vision features after pixel shuffle. *Shape:* ``[B, N, encoder_dim]``.

        :returns:
            Projected features. *Shape:* ``[B, N, model_dim]``.
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
