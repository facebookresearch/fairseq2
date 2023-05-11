# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import AbstractSet, Final, Optional

import torch

from fairseq2.data.text import VocabularyInfo
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerModel,
)
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.position_encoder import SinusoidalPositionEncoder
from fairseq2.nn.projection import TiedProjection
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerNormOrder,
    get_default_sdpa,
)


@dataclass
class NllbConfig:
    """Holds the configuration of an NLLB model."""

    max_seq_len: int = 1024
    """The expected maximum sequence length."""

    model_dim: int = 1024
    """The dimensionality of the model."""

    num_encoder_layers: int = 24
    """The number of Transformer encoder layers."""

    num_decoder_layers: int = 24
    """The number of Transformer decoder layers."""

    num_encoder_attn_heads: int = 16
    """The number of attention heads in Transformer encoder layers."""

    num_decoder_attn_heads: int = 16
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int = 1024 * 8
    """The dimensionality of inner projection layers in Transformer feed-forward
    networks."""

    dropout_p: float = 0.1
    """The dropout probability in Transformer layers."""


_CONFIGS: Final = {
    "dense_1b": lambda: NllbConfig(
        model_dim=1024,
        num_encoder_layers=24,
        num_decoder_layers=24,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    ),
    "dense_3b": lambda: NllbConfig(
        model_dim=2048,
        num_encoder_layers=24,
        num_decoder_layers=24,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    ),
    "dense_600m": lambda: NllbConfig(
        model_dim=1024,
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 4,
        dropout_p=0.1,
    ),
}


def get_nllb_archs() -> AbstractSet[str]:
    """Return the names of supported NLLB architectures."""
    return _CONFIGS.keys()


def get_nllb_config(arch_name: str) -> NllbConfig:
    """Return the configuration of the specified NLLB architecture.

    :param arch_name:
        The name of the architecture.
    """
    try:
        return _CONFIGS[arch_name]()
    except KeyError:
        raise ValueError(
            f"`arch_name` must be a known NLLB architecture, but is '{arch_name}' instead."
        )


def create_nllb_model(
    cfg: NllbConfig,
    vocab_info: VocabularyInfo,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> TransformerModel:
    """Create an NLLB model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    :param cfg:
        The configuration to use.
    :param tokenizer:
        The vocabulary information to use.
    :param device:
        The device on which to initialize the model.
    :param dtype:
        The data type of the model parameters and buffers.
    """
    return NllbBuilder(cfg, vocab_info, device, dtype).build_model()


class NllbBuilder:
    """Builds modules of an NLLB model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    To tweak the model architecture, you can derive from this class and override
    the corresponding methods.
    """

    cfg: NllbConfig
    vocab_info: VocabularyInfo
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    def __init__(
        self,
        cfg: NllbConfig,
        vocab_info: VocabularyInfo,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param cfg:
            The configuration to use.
        :param vocab_info:
            The vocabulary information to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.cfg = cfg
        self.vocab_info = vocab_info
        self.device = device
        self.dtype = dtype

    def build_model(self) -> TransformerModel:
        """Build a model."""
        embed = Embedding(
            num_embeddings=self.vocab_info.size,
            embedding_dim=self.cfg.model_dim,
            pad_idx=self.vocab_info.pad_idx,
            scaled=True,
            device=self.device,
            dtype=self.dtype,
        )

        frontend = self.build_frontend(embed)

        encoder = self.build_encoder()
        decoder = self.build_decoder()

        final_proj = TiedProjection(embed.weight)

        return TransformerModel(
            frontend, encoder, frontend, decoder, final_proj, self.vocab_info.pad_idx
        )

    def build_frontend(self, embed: Embedding) -> TransformerFrontend:
        """Build a shared Transformer encoder/decoder front-end."""
        pos_encoder = SinusoidalPositionEncoder(
            self.cfg.model_dim,
            self.cfg.max_seq_len,
            _legacy_pad_idx=self.vocab_info.pad_idx,
            device=self.device,
            dtype=self.dtype,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        layers = [
            self.build_encoder_layer() for _ in range(self.cfg.num_encoder_layers)
        ]

        return StandardTransformerEncoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        layers = [
            self.build_decoder_layer() for _ in range(self.cfg.num_decoder_layers)
        ]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        self_attn = self.build_attention(self.cfg.num_encoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(self.cfg.num_decoder_attn_heads)

        encoder_decoder_attn = self.build_attention(self.cfg.num_decoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = get_default_sdpa(attn_dropout_p=self.cfg.dropout_p)

        return StandardMultiheadAttention(
            num_heads,
            self.cfg.model_dim,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.cfg.model_dim,
            self.cfg.ffn_inner_dim,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )
