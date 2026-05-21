# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen 3.6 ViT vision encoder.

Architecture (from HF checkpoint weights):
- Conv3d patch embedding: (3 -> hidden_size, kernel=(temporal_patch_size, patch_size, patch_size))
- Learned position embeddings: (num_position_embeddings, hidden_size)
- N x QwenVisionBlock: LayerNorm + fused QKV attention + LayerNorm + MLP (GELU)
- QwenVisionMerger: RMSNorm + spatial 2x2 merge + 2-layer MLP -> out_hidden_size
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq2.models.qwen.config import Qwen36VisionConfig


class QwenVisionBlock(nn.Module):
    """Single ViT transformer block with fused QKV attention."""

    num_heads: Final[int]
    head_dim: Final[int]

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Pre-norm + fused QKV attention
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = _VisionAttention(hidden_size, num_heads)

        # Pre-norm + MLP
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = _VisionMLP(hidden_size, intermediate_size)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # Pre-norm attention + residual
        x = x + self.attn(self.norm1(x), attention_mask)
        # Pre-norm MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


class _VisionAttention(nn.Module):
    """Fused QKV multi-head attention for vision blocks."""

    num_heads: Final[int]
    head_dim: Final[int]

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Fused Q+K+V projection: (hidden_size -> 3 * hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        B, S, _ = x.shape

        # (B, S, 3*H) -> 3 x (B, S, num_heads, head_dim)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # (B, S, num_heads, head_dim) -> (B, num_heads, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        # (B, num_heads, S, head_dim) -> (B, S, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)

        return self.proj(attn_output)


class _VisionMLP(nn.Module):
    """MLP with GELU(tanh) activation for vision blocks."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()

        # HF checkpoint uses linear_fc1 / linear_fc2 naming
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class QwenVisionEncoder(nn.Module):
    """Full vision encoder: patch embed + pos embed + N blocks."""

    def __init__(self, config: Qwen36VisionConfig) -> None:
        super().__init__()

        self.config = config

        # Conv3d patch embedding
        self.patch_embed = _PatchEmbed(
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
        )

        # Learned position embeddings
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            QwenVisionBlock(config.hidden_size, config.num_heads, config.intermediate_size)
            for _ in range(config.depth)
        ])

    def forward(
        self,
        pixel_values: Tensor,
        grid_thw: Tensor,
    ) -> Tensor:
        """
        Args:
            pixel_values: (num_patches, C * temporal_patch_size * patch_size * patch_size)
            grid_thw: (num_images, 3) — (temporal, height, width) grid sizes per image.

        Returns:
            Vision hidden states: (num_patches, hidden_size)
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)

        # Position IDs and embedding
        pos_ids = self._compute_position_ids(grid_thw)
        hidden_states = hidden_states + self.pos_embed(pos_ids)

        # Reshape for transformer blocks: (1, total_patches, hidden_size)
        hidden_states = hidden_states.unsqueeze(0)

        # Vision attention mask (block-diagonal per image)
        attention_mask = self._compute_attention_mask(grid_thw, hidden_states.device)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        # Remove batch dim: (total_patches, hidden_size)
        return hidden_states.squeeze(0)

    def _compute_position_ids(self, grid_thw: Tensor) -> Tensor:
        """Compute position IDs for 2D spatial positions within each image."""
        pos_ids_list = []
        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()
            # Create 2D position grid
            hpos = torch.arange(h, device=grid_thw.device)
            wpos = torch.arange(w, device=grid_thw.device)
            grid = hpos[:, None] * w + wpos[None, :]  # (h, w)
            # Repeat for temporal dimension
            pos_ids = grid.reshape(-1).repeat(t)
            pos_ids_list.append(pos_ids)
        return torch.cat(pos_ids_list)

    def _compute_attention_mask(self, grid_thw: Tensor, device: torch.device) -> Tensor | None:
        """Compute block-diagonal attention mask: each image attends only to itself."""
        if grid_thw.shape[0] == 1:
            return None  # Single image, no mask needed

        # Compute per-image patch counts
        patch_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        total_patches = sum(patch_counts)

        # Build block-diagonal mask
        mask = torch.full(
            (1, 1, total_patches, total_patches),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        start = 0
        for count in patch_counts:
            mask[:, :, start : start + count, start : start + count] = 0.0
            start += count

        return mask


class _PatchEmbed(nn.Module):
    """Conv3d-based patch embedding for images/video."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
    ) -> None:
        super().__init__()

        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Args:
            pixel_values: Flattened patches (num_patches, C * t * p * p).
                For the standard config: each patch is 3 * 2 * 16 * 16 = 1536 values.

        Returns:
            (num_patches, hidden_size)
        """
        # Reconstruct 5D tensor for Conv3d
        # pixel_values: (N, C*T*P*P)
        target_dtype = self.proj.weight.dtype
        pixel_values = pixel_values.to(dtype=target_dtype)

        # Each patch vector: C * temporal_patch_size * patch_size * patch_size
        C = self.proj.in_channels
        T = self.proj.kernel_size[0]
        P = self.proj.kernel_size[1]

        N = pixel_values.shape[0]
        # (N, C, T, P, P)
        x = pixel_values.reshape(N, C, T, P, P)

        # Conv3d: (N, C, T, P, P) -> (N, hidden_size, 1, 1, 1)
        x = self.proj(x)

        # (N, hidden_size, 1, 1, 1) -> (N, hidden_size)
        return x.flatten(1)


class QwenVisionMerger(nn.Module):
    """Merges 2x2 neighboring vision patches and projects to text dimension.

    Pipeline:
    1. RMSNorm on vision hidden states
    2. Spatial merge: group 2x2 neighbors -> concatenate -> 4 * hidden_size
    3. fc1: Linear(4 * hidden_size -> 4 * hidden_size) + GELU
    4. fc2: Linear(4 * hidden_size -> out_hidden_size)
    """

    def __init__(self, config: Qwen36VisionConfig) -> None:
        super().__init__()

        self.spatial_merge_size = config.spatial_merge_size
        hidden_size = config.hidden_size
        merged_dim = hidden_size * config.spatial_merge_size ** 2

        # NOTE: HF checkpoint uses RMSNorm with bias for merger
        self.norm = _RMSNormWithBias(hidden_size)
        self.linear_fc1 = nn.Linear(merged_dim, merged_dim, bias=True)
        self.linear_fc2 = nn.Linear(merged_dim, config.out_hidden_size, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        grid_thw: Tensor,
    ) -> Tensor:
        """
        Args:
            hidden_states: (total_patches, hidden_size)
            grid_thw: (num_images, 3) — grid sizes per image

        Returns:
            (total_merged_patches, out_hidden_size)
        """
        hidden_states = self.norm(hidden_states)

        merged_list = []
        offset = 0
        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()
            num_patches = t * h * w
            img_states = hidden_states[offset : offset + num_patches]  # (t*h*w, D)
            offset += num_patches

            # Reshape to spatial: (t, h, w, D)
            img_states = img_states.reshape(t, h, w, -1)

            # Spatial merge: group 2x2 patches
            s = self.spatial_merge_size
            h_out, w_out = h // s, w // s
            # (t, h_out, s, w_out, s, D) -> (t, h_out, w_out, s, s, D)
            img_states = img_states.reshape(t, h_out, s, w_out, s, -1)
            img_states = img_states.permute(0, 1, 3, 2, 4, 5)  # (t, h_out, w_out, s, s, D)
            # Flatten merge groups: (t * h_out * w_out, s*s*D)
            img_states = img_states.reshape(t * h_out * w_out, -1)

            merged_list.append(img_states)

        merged = torch.cat(merged_list, dim=0)  # (total_merged_patches, s*s*D)

        # MLP: fc1 + GELU + fc2
        merged = F.gelu(self.linear_fc1(merged), approximate="tanh")
        merged = self.linear_fc2(merged)

        return merged


class _RMSNormWithBias(nn.Module):
    """RMSNorm with an optional bias term (used by Qwen 3.6 vision merger)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x + self.bias).to(input_dtype)
