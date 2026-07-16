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

    Much simpler than Gemma3n's embedder -- no hard/soft token distinction,
    no embedding lookup table.  Applies ``RMSNorm`` (without learnable scale)
    followed by a ``Linear`` projection from ``output_proj_dims`` to
    ``text_model_dim``.

    This single class implements **two** matching HF classes:

    * ``transformers.Gemma4MultimodalEmbedder`` (classic gemma4 family,
      E4B/31B/26B-A4B): forward is ``RMSNorm -> Linear``, callers are
      expected to feed inputs already in the embedder's dtype.

    * ``transformers.Gemma4UnifiedMultimodalEmbedder`` (gemma4_unified
      family, 12B+): identical math, but the forward additionally casts
      ``inputs_embeds`` to ``self.embedding_projection.weight.dtype`` before
      the norm. This matters when raw waveform features (typically fp32 from
      the feature extractor) are fed into a bf16 embedder.

    The ``cast_input_dtype`` ctor flag selects between the two: ``False``
    (default) preserves the classic gemma4 behaviour bit-for-bit; ``True``
    activates the Unified family's input cast. The factory sets it to
    ``True`` when ``audio_config.audio_mode == "linear"``.

    Note: HF does NOT use ClippableLinear for the embedder projection --
    the checkpoint key is ``model.embed_audio.embedding_projection.weight``
    (plain ``nn.Linear``, no clipping buffers).
    """

    embedding_pre_projection_norm: Gemma4AudioRMSNorm
    embedding_projection: Linear
    cast_input_dtype: bool

    def __init__(
        self,
        output_proj_dims: int,
        text_model_dim: int,
        rms_norm_eps: float = 1e-6,
        *,
        cast_input_dtype: bool = False,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        super().__init__()

        self.cast_input_dtype = cast_input_dtype

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
        if self.cast_input_dtype:
            # Match HF Gemma4UnifiedMultimodalEmbedder: cast raw inputs
            # (often fp32 from the feature extractor) to the embedder weight
            # dtype (typically bf16) before the norm.
            features = features.to(self.embedding_projection.weight.dtype)
        features = self.embedding_pre_projection_norm(features)
        features = self.embedding_projection(features)
        return features
