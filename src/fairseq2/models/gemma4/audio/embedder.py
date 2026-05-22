# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.gemma4.audio.norm import Gemma4AudioRMSNorm
from fairseq2.nn.projection import Linear


@final
class Gemma4MultimodalAudioEmbedder(Module):
    """Projects audio tower output to text model space.

    Much simpler than Gemma3n's embedder — no hard/soft token distinction,
    no embedding lookup table. Just:
      RMSNorm(output_proj_dims, with_scale=False) -> Linear(output_proj_dims, text_model_dim)

    Note: HF does NOT use ClippableLinear for the embedder projection —
    the checkpoint key is ``model.embed_audio.embedding_projection.weight``
    (plain nn.Linear, no clipping buffers).
    """

    embedding_pre_projection_norm: Gemma4AudioRMSNorm
    embedding_projection: Linear

    def __init__(
        self,
        output_proj_dims: int,
        text_model_dim: int,
        rms_norm_eps: float = 1e-6,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        # RMSNorm without learnable scale (elementwise_affine=False)
        self.embedding_pre_projection_norm = Gemma4AudioRMSNorm(
            output_proj_dims,
            bias=False,
            eps=rms_norm_eps,
            elementwise_affine=False,
            device=device,
            dtype=dtype,
        )

        self.embedding_projection = Linear(
            output_proj_dims,
            text_model_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(self, features: Tensor) -> Tensor:
        """
        :param features: Audio tower output. *Shape:* :math:`(N,T,D)`.
        :returns: Text-space embeddings. *Shape:* :math:`(N,T,H_{text})`.
        """
        features = self.embedding_pre_projection_norm(features)
        features = self.embedding_projection(features)
        return features
