# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NemotronH 3-way hybrid decoder layer.

Each layer in NemotronH is SINGLE-PURPOSE — it is either:
- Mamba2 SSM (23 layers)
- MoE FFN (23 layers)
- Full GQA Attention (6 layers)

This is fundamentally different from standard Transformer layers which have
both attention AND FFN in each layer.

The layer applies: pre-norm (RMSNorm) -> mixer -> residual
"""

from __future__ import annotations

import math
from typing import final

from torch import Tensor
from typing_extensions import override

from fairseq2.models.nemotron.config import BlockType
from fairseq2.models.nemotron.mamba2 import NemotronHMamba2Mixer
from fairseq2.models.nemotron.moe import NemotronHMoE
from fairseq2.models.transformer import (
    AttentionBiasCache,
    MultiheadAttention,
)
from fairseq2.models.transformer_lm.decoder_layer import TransformerLMDecoderLayer
from fairseq2.nn import BatchLayout, IncrementalStateBag, LayerNorm


@final
class NemotronHBlock(TransformerLMDecoderLayer):
    """A single-purpose decoder block for NemotronH.

    Depending on block_type, the mixer is one of:
    - NemotronHMamba2Mixer (for "mamba" blocks)
    - MultiheadAttention (for "attention" blocks)
    - NemotronHMoE (for "moe" blocks)

    Forward: pre_norm(x) -> mixer -> residual + x
    """

    def __init__(
        self,
        block_type: BlockType,
        mixer: NemotronHMamba2Mixer | MultiheadAttention | NemotronHMoE,
        norm: LayerNorm,
        *,
        layer_idx: int = 0,
        num_layers: int = 52,
        rescale_prenorm_residual: bool = True,
    ) -> None:
        super().__init__()

        self.block_type = block_type
        self.mixer = mixer
        self.norm = norm
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.rescale_prenorm_residual = rescale_prenorm_residual

        # GPT-2 style residual scaling: 1/sqrt(2*num_layers)
        if rescale_prenorm_residual:
            self._residual_scale = 1.0 / math.sqrt(2 * num_layers)
        else:
            self._residual_scale = 1.0

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        attn_bias_cache: AttentionBiasCache,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """Forward pass dispatching to the appropriate mixer.

        Args:
            seqs: Input sequences. Shape: [B, S, M]
            seqs_layout: Batch layout information.
            attn_bias_cache: Attention bias cache (used only by attention blocks).
            state_bag: Incremental state bag for generation.

        Returns:
            Output sequences. Shape: [B, S, M]
        """
        residual = seqs

        # Pre-normalization
        seqs = self.norm(seqs)

        # Dispatch to appropriate mixer
        if self.block_type == "mamba":
            assert isinstance(self.mixer, NemotronHMamba2Mixer)
            seqs = self.mixer(seqs, state_bag=state_bag)

        elif self.block_type == "attention":
            assert isinstance(self.mixer, MultiheadAttention)
            seqs = self.mixer(
                seqs,
                seqs_layout,
                keys=seqs,
                keys_layout=seqs_layout,
                values=seqs,
                bias_cache=attn_bias_cache,
                state_bag=state_bag,
            )

        elif self.block_type == "moe":
            assert isinstance(self.mixer, NemotronHMoE)
            seqs = self.mixer(seqs)

        else:
            raise ValueError(f"Unknown block type: {self.block_type}")

        # Residual connection (no runtime scaling — HF bakes the scaling
        # into out_proj weight init via _init_weights, not at runtime)
        seqs = seqs + residual

        return seqs

    @override
    def extra_repr(self) -> str:
        return (
            f"block_type={self.block_type}, "
            f"layer_idx={self.layer_idx}, "
            f"rescale={self.rescale_prenorm_residual}"
        )
