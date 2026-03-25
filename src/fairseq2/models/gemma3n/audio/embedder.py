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
from fairseq2.models.gemma3n.config import Gemma3nAudioConfig, Gemma3nConfig
from fairseq2.nn import RMSNorm, StandardEmbedding
from fairseq2.nn.projection import Linear


@final
class Gemma3nMultimodalEmbedder(Module):
    """Projects audio features to text model space.

    Handles both hard audio tokens (discrete IDs) and soft audio embeddings
    (continuous features from conformer encoder). Projects from audio hidden
    size (1536) to text model hidden size (2048).
    """

    embedding: StandardEmbedding
    hard_embedding_norm: RMSNorm
    soft_embedding_norm: RMSNorm
    embedding_projection: Linear
    embedding_post_projection_norm: RMSNorm

    def __init__(
        self,
        audio_config: Gemma3nAudioConfig,
        text_config: Gemma3nConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param audio_config: Audio tower configuration.
        :param text_config: Text model configuration.
        """
        super().__init__()

        self.embedding = StandardEmbedding(
            num_embeddings=audio_config.vocab_size,
            embed_dim=audio_config.hidden_size,
            device=device,
            dtype=dtype,
        )

        self.hard_embedding_norm = RMSNorm(
            audio_config.hidden_size,
            bias=False,
            eps=audio_config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.soft_embedding_norm = RMSNorm(
            audio_config.hidden_size,
            bias=False,
            eps=audio_config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.embedding_projection = Linear(
            audio_config.hidden_size,
            text_config.model_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.embedding_post_projection_norm = RMSNorm(
            text_config.model_dim,
            bias=False,
            eps=text_config.rms_norm_eps,
            elementwise_affine=False,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(self, features: Tensor, is_soft: bool = True) -> Tensor:
        """
        :param features: Either soft embeddings from conformer *Shape:* :math:`(N,T,H_audio)`
                        or hard token IDs *Shape:* :math:`(N,T)` where H_audio=1536.
        :param is_soft: If True, treat features as soft embeddings from encoder.
                       If False, treat as hard token IDs for embedding lookup.
        :returns: Projected features. *Shape:* :math:`(N,T,H_text)` where H_text=2048.
        """
        if is_soft:
            seqs = self.soft_embedding_norm(features)
        else:
            seqs = self.embedding(features)
            seqs = self.hard_embedding_norm(seqs)

        seqs = self.embedding_projection(seqs)

        seqs = self.embedding_post_projection_norm(seqs)

        return seqs
