# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from fairseq2.models.llama4.config import LLaMA4VisionEncoderConfig
from fairseq2.models.llama4.model.vision.encoder import VisionEncoder
from fairseq2.models.llama4.model.vision.ffn import _FeedForward
from fairseq2.nn import Linear


class PixelShuffle(nn.Module):
    def __init__(self, ps_ratio: float) -> None:
        super().__init__()
        self.ps_ratio = ps_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C], N = number of patches
        assert self.ps_ratio is not None, "ps_ratio is required for pixel shuffle"
        assert x.dim() == 3, "pixel shuffle requires encoded patches [B, N, C]"
        hh = ww = int(math.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], hh, ww, -1)
        x = pixel_shuffle_op(x, ps_ratio=self.ps_ratio)
        pixel_shuffle_patches = x.reshape(x.shape[0], -1, x.shape[-1])
        return pixel_shuffle_patches


def pixel_shuffle_op(input_x: torch.Tensor, ps_ratio: float) -> torch.Tensor:
    n, w, h, c = input_x.size()
    input_x = input_x.view(n, w, int(h * ps_ratio), int(c / ps_ratio))
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    input_x = input_x.view(
        n,
        int(h * ps_ratio),
        int(w * ps_ratio),
        int(c / (ps_ratio * ps_ratio)),
    )
    input_x = input_x.permute(0, 2, 1, 3).contiguous()
    return input_x


class PixelShuffleMLP(torch.nn.Module):
    def __init__(
        self,
        ps_ratio: float,
        input_dim: int,
        output_dim: int = 4096,
        add_fc: bool = False,
    ) -> None:
        super().__init__()
        self.pixel_shuffle = PixelShuffle(ps_ratio)
        self.mlp = _FeedForward(
            int(input_dim // (ps_ratio**2)),
            output_dim,
            bias=False,
            dropout=0.0,
            act_layer=nn.GELU,
            act_on_output=True,
        )
        self.fc = nn.Identity()
        if add_fc:
            # in the reference code: self.fc = ColumnParallelLinear(
            # but this is not used atm
            Linear(
                output_dim,
                output_dim,
                bias=False,
            )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = self.pixel_shuffle(encoded_patches)
        return self.fc(self.mlp(encoded_patches))  # type: ignore


class VisionEmbeddings(torch.nn.Module):
    """
    The Llama 4 vision embedding class has two main parts:
    1. A transformer encoder, which could be frozen during finetuning
    2. An adapter, which should be always be trained.
    """

    def __init__(self, config: LLaMA4VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config

        image_size = config.image_size
        patch_size = config.patch_size
        self.vision_encoder = VisionEncoder(
            image_size=(image_size, image_size),
            patch_size=(patch_size, patch_size),
            dim=config.model_dim,
            layers=config.num_layers,
            heads=config.num_attn_heads,
            mlp_ratio=config.mlp_ratio,
        )
        self.vision_encoder = self.vision_encoder.to(torch.bfloat16)
        self.vision_adapter = PixelShuffleMLP(
            ps_ratio=config.pixel_shuffle_ratio,
            input_dim=config.model_dim,
            output_dim=config.output_dim,
        )

        self.output_dim = config.output_dim
        # NOTE(mg): should not be needed
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool = True,
        missing_keys: Optional[List[str]] = None,
        unexpected_keys: Optional[List[str]] = None,
        error_msgs: Optional[List[str]] = None,
        return_state_dict: bool = False,
    ) -> None:
        # Should not be useful
        original_sd = self.state_dict()
        for k in state_dict:
            if (
                k.startswith(prefix)
                and len(state_dict[k].shape) == 1
                and state_dict[k].shape[0] == 0
            ):
                state_dict[k] = state_dict[k].reshape(
                    original_sd[k[len(prefix) :]].shape
                )

    def _get_empty_sequence(self, h: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            h.shape[0],
            h.shape[1],
            self.output_dim,
            device=h.device,
            dtype=h.dtype,
        )

    # x_images is batched; each batch sample contains a list of images. so this is List[List[torch.Tensor]]
    # each image is a tensor of shape [num_tiles, C, H, W]
    def forward(
        self,
        image_batch: List[List[torch.Tensor]],
        image_mask: torch.Tensor,
        h_ref: torch.Tensor,
    ) -> torch.Tensor:
        images_flattened = [image for sample in image_batch for image in sample]
        images_tensor = (
            torch.vstack(images_flattened).unsqueeze(1).to(h_ref.dtype).to(h_ref.device)
        )
        embedding = self.vision_encoder(images_tensor)
        projected_embedding = self.vision_adapter(embedding)

        h_image = self._get_empty_sequence(h_ref)
        return scatter_embeddings(image_batch, image_mask, h_image, projected_embedding)


def scatter_embeddings(
    image_batch: List[List[torch.Tensor]],
    image_mask: torch.Tensor,
    h_image: torch.Tensor,
    encoded_patches_proj: torch.Tensor,
) -> torch.Tensor:
    # If dynamic transform is used and the batch contains 2 images (where image_1 has 2 chunks and image_2 has 3 chunks),
    # `num_images_per_sequence` now records the number of chunks per image as `[2, 3]`.
    # `encoded_patches_proj.split` will then split the image chunks into 2 groups: `[image_1_chunks, image_2_chunks]`.
    num_images_per_sequence = [
        sum(image.size(0) for image in sample_images) for sample_images in image_batch
    ]

    assert not torch.isnan(encoded_patches_proj).any()
    assert sum(num_images_per_sequence) == encoded_patches_proj.size(
        0
    ), f"{sum(num_images_per_sequence)=} != {encoded_patches_proj.shape=}"

    encoded_patches_list = encoded_patches_proj.split(num_images_per_sequence, dim=0)
    for index in range(h_image.size(0)):
        encoded_patches_per_sample = encoded_patches_list[index]
        sample_image_mask = image_mask[index]

        if encoded_patches_per_sample.numel() == 0:
            continue
        encoded_patches_per_sample = encoded_patches_per_sample.contiguous().view(
            -1, encoded_patches_per_sample.size(-1)
        )

        n_tokens_to_fill = sample_image_mask.sum()
        assert n_tokens_to_fill <= encoded_patches_per_sample.size(0)

        h_image[index].masked_scatter_(
            sample_image_mask.expand(-1, h_image.size(-1)),
            encoded_patches_per_sample[:n_tokens_to_fill],
        )

    return h_image
