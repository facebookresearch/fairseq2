# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""C-RADIO ViT encoder for the NemotronH vision tower.

Implements a ViT-Huge encoder matching NVIDIA's C-RADIO (Comprehensive
RADio-Improved Observations) architecture:

  - Linear patch embedding (flatten 16×16×3 → 768 → Linear → 1280)
  - 10 learnable register tokens (cls_token in HF, but actually registers)
  - Interpolated position embeddings supporting arbitrary resolutions
  - 32 transformer blocks with fused QKV attention, GELU MLP, LayerNorm
  - Input conditioner for image normalization (norm_mean/norm_std buffers)

The encoder outputs per-patch features (excluding register tokens) that are
passed through pixel_shuffle + VisionProjection to produce LM-space embeddings.

HF state dict key structure:
    vision_model.radio_model.input_conditioner.norm_{mean,std}
    vision_model.radio_model.model.patch_generator.cls_token.token  [10, 1280]
    vision_model.radio_model.model.patch_generator.embedder.weight  [1280, 768]
    vision_model.radio_model.model.patch_generator.pos_embed  [1, 16384, 1280]
    vision_model.radio_model.model.blocks.{i}.attn.qkv.{weight,bias}
    vision_model.radio_model.model.blocks.{i}.attn.proj.{weight,bias}
    vision_model.radio_model.model.blocks.{i}.mlp.fc1.{weight,bias}
    vision_model.radio_model.model.blocks.{i}.mlp.fc2.{weight,bias}
    vision_model.radio_model.model.blocks.{i}.norm{1,2}.{weight,bias}
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor
from torch.nn import GELU, LayerNorm, Linear, Module, ModuleList, Parameter
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device


@final
class CRADIOViTBlock(Module):
    """A single ViT transformer block for C-RADIO.

    Architecture:
        norm1 → fused_qkv_attention → residual
        norm2 → MLP(fc1→GELU→fc2) → residual

    Uses standard LayerNorm with bias (not RMSNorm), fused QKV projection,
    and GELU activation in MLP (not SiLU/SquaredReLU).
    """

    def __init__(
        self,
        hidden_size: int = 1280,
        num_heads: int = 16,
        mlp_dim: int = 5120,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Pre-norms (LayerNorm with bias=True)
        self.norm1 = LayerNorm(hidden_size, device=device, dtype=dtype)
        self.norm2 = LayerNorm(hidden_size, device=device, dtype=dtype)

        # Fused QKV attention
        self.attn_qkv = Linear(
            hidden_size, 3 * hidden_size, bias=True, device=device, dtype=dtype
        )
        self.attn_proj = Linear(
            hidden_size, hidden_size, bias=True, device=device, dtype=dtype
        )

        # MLP
        self.mlp_fc1 = Linear(
            hidden_size, mlp_dim, bias=True, device=device, dtype=dtype
        )
        self.mlp_fc2 = Linear(
            mlp_dim, hidden_size, bias=True, device=device, dtype=dtype
        )
        self.mlp_act = GELU()

        self._scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through one ViT block.

        :param x:
            Input features. *Shape:* ``[B, N, D]`` where N = num_patches + num_registers.

        :returns:
            Output features. *Shape:* ``[B, N, D]``.
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self._attention(x)
        x = residual + x

        # MLP with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)
        x = residual + x

        return x

    def _attention(self, x: Tensor) -> Tensor:
        """Fused QKV multi-head self-attention.

        :param x:
            Normed input. *Shape:* ``[B, N, D]``.

        :returns:
            Attention output. *Shape:* ``[B, N, D]``.
        """
        b, n, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        # Fused QKV: [B, N, D] -> [B, N, 3*D] -> [B, N, 3, H, d]
        qkv = self.attn_qkv(x).reshape(b, n, 3, h, d)

        # [B, N, 3, H, d] -> [3, B, H, N, d]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, N, d]

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self._scale
        attn = torch.softmax(attn, dim=-1)

        # [B, H, N, d] -> [B, N, H, d] -> [B, N, D]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, self.hidden_size)

        return self.attn_proj(out)

    if TYPE_CHECKING:
        __call__ = forward

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )


@final
class CRADIOViTEncoder(Module):
    """C-RADIO ViT-Huge vision encoder for NemotronH.

    Pipeline:
        image [B, 3, H, W]
        → input conditioner (normalize with norm_mean/norm_std)
        → patch embedding (im2patches → Linear)
        → prepend register tokens
        → add interpolated position embeddings
        → N transformer blocks
        → strip register tokens
        → output [B, num_patches, hidden_size]

    Register tokens (10 in C-RADIO, stored as ``cls_token`` in HF but
    functioning as registers) are prepended before position encoding
    and stripped after the final block.

    Position embeddings are stored at max resolution (128×128 = 16384 patches)
    and bilinearly interpolated for other resolutions.
    """

    def __init__(
        self,
        hidden_size: int = 1280,
        num_heads: int = 16,
        num_layers: int = 32,
        mlp_dim: int = 5120,
        patch_size: int = 16,
        num_registers: int = 10,
        max_grid_size: int = 128,
        image_size: int = 512,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_registers = num_registers
        self.max_grid_size = max_grid_size
        self.image_size = image_size

        # Input conditioner: normalization buffers
        self.norm_mean = torch.nn.Parameter(
            torch.zeros(3, 1, 1, device=device, dtype=dtype), requires_grad=False
        )
        self.norm_std = torch.nn.Parameter(
            torch.ones(3, 1, 1, device=device, dtype=dtype), requires_grad=False
        )

        # Patch embedding: Linear(patch_size^2 * 3, hidden_size)
        # C-RADIO uses Im2Patches + Linear, not Conv2d
        # HF key: patch_generator.embedder.weight [1280, 768]
        patch_dim = patch_size * patch_size * 3  # 16*16*3 = 768
        self.patch_embed = Linear(
            patch_dim, hidden_size, bias=False, device=device, dtype=dtype
        )

        # Register (cls) tokens: [num_registers, hidden_size]
        # HF key: patch_generator.cls_token.token [10, 1280]
        self.cls_token = Parameter(
            torch.zeros(num_registers, hidden_size, device=device, dtype=dtype)
        )

        # Position embeddings: [1, max_grid_size^2, hidden_size]
        # HF key: patch_generator.pos_embed [1, 16384, 1280]
        max_num_patches = max_grid_size * max_grid_size
        self.pos_embed = Parameter(
            torch.zeros(1, max_num_patches, hidden_size, device=device, dtype=dtype)
        )

        # Video embedder exists in HF but we skip it (vision-only, not video)
        # HF key: patch_generator.video_embedder.weight [1280, 1536]
        # This is loaded but never used in image-only mode.
        self.video_embedder = Linear(
            patch_dim * 2, hidden_size, bias=False, device=device, dtype=dtype
        )

        # Transformer blocks
        self.blocks = ModuleList(
            [
                CRADIOViTBlock(
                    hidden_size, num_heads, mlp_dim,
                    device=device, dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Encode image into per-patch features.

        :param pixel_values:
            Images. *Shape:* ``[B, 3, H, W]``.

        :returns:
            Per-patch features (registers stripped).
            *Shape:* ``[B, num_patches, hidden_size]``.
        """
        b, c, h, w = pixel_values.shape
        p = self.patch_size

        # Input conditioning: normalize
        x = (pixel_values - self.norm_mean) / self.norm_std

        # Im2Patches: [B, 3, H, W] -> [B, H/p, W/p, p*p*3] -> [B, N, patch_dim]
        gh = h // p  # grid height
        gw = w // p  # grid width
        num_patches = gh * gw

        # Reshape to extract patches
        # [B, 3, H, W] -> [B, 3, gh, p, gw, p]
        x = x.reshape(b, c, gh, p, gw, p)
        # [B, 3, gh, p, gw, p] -> [B, gh, gw, 3, p, p]
        x = x.permute(0, 2, 4, 1, 3, 5)
        # [B, gh, gw, 3*p*p] -> [B, N, patch_dim]
        x = x.reshape(b, num_patches, c * p * p)

        # Linear patch embedding: [B, N, patch_dim] -> [B, N, hidden_size]
        x = self.patch_embed(x)

        # Prepend register tokens: [B, num_registers + N, hidden_size]
        cls_tokens = self.cls_token.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add interpolated position embeddings (only to patch tokens, not registers)
        pos = self._interpolate_pos_embed(gh, gw)  # [1, N, hidden_size]
        x[:, self.num_registers:] = x[:, self.num_registers:] + pos

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Strip register tokens: [B, N, hidden_size]
        x = x[:, self.num_registers:]

        return x

    def _interpolate_pos_embed(self, grid_h: int, grid_w: int) -> Tensor:
        """Interpolate position embeddings for the given grid size.

        Position embeddings are stored as a flat sequence for max_grid_size^2 patches.
        For other resolutions, bilinear interpolation is used.

        :param grid_h:
            Number of patches in height.
        :param grid_w:
            Number of patches in width.

        :returns:
            Interpolated position embeddings. *Shape:* ``[1, grid_h * grid_w, hidden_size]``.
        """
        num_patches = grid_h * grid_w

        if grid_h == self.max_grid_size and grid_w == self.max_grid_size:
            return self.pos_embed[:, :num_patches]

        # Reshape stored pos_embed from flat to 2D grid
        # [1, max_grid^2, D] -> [1, D, max_grid, max_grid]
        pos = self.pos_embed.reshape(
            1, self.max_grid_size, self.max_grid_size, self.hidden_size
        ).permute(0, 3, 1, 2)

        # Bilinear interpolation to target grid size
        pos = torch.nn.functional.interpolate(
            pos.float(),
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=self.pos_embed.dtype)

        # [1, D, grid_h, grid_w] -> [1, grid_h * grid_w, D]
        pos = pos.permute(0, 2, 3, 1).reshape(1, num_patches, self.hidden_size)

        return pos

    if TYPE_CHECKING:
        __call__ = forward

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_layers={len(self.blocks)}, "
            f"patch_size={self.patch_size}, "
            f"num_registers={self.num_registers}"
        )
