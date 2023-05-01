# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import AbstractSet, Final, Optional

import torch

from fairseq2.data.text import VocabularyInfo
from fairseq2.models.encoder_decoder import EncoderDecoderFrontend
from fairseq2.models.transformer import TransformerModel, TransformerTokenFrontend
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.positional_encoder import SinusoidalPositionalEncoder
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
)


@dataclass
class NllbConfig:
    """Holds the configuration of an NLLB model."""

    max_seq_len: int = 1024
    """The expected maximum sequence length."""

    model_dim: int = 1024
    """The dimensionality of the model."""

    num_enc_layers: int = 24
    """The number of encoder layers."""

    num_dec_layers: int = 24
    """The number of decoder layers."""

    num_enc_attn_heads: int = 16
    """The number of attention heads in encoder layers."""

    num_dec_attn_heads: int = 16
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 1024 * 8
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of embedding dictionaries, attention
    layers, and feed-forward networks."""

    dtype: torch.dtype = torch.float32
    """The data type of model parameters and buffers."""


_CONFIGS: Final = {
    "dense_1b": lambda: NllbConfig(
        model_dim=1024,
        num_enc_layers=24,
        num_dec_layers=24,
        num_enc_attn_heads=16,
        num_dec_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    ),
    "dense_3b": lambda: NllbConfig(
        model_dim=2048,
        num_enc_layers=24,
        num_dec_layers=24,
        num_enc_attn_heads=16,
        num_dec_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    ),
    "dense_600m": lambda: NllbConfig(
        model_dim=1024,
        num_enc_layers=12,
        num_dec_layers=12,
        num_enc_attn_heads=16,
        num_dec_attn_heads=16,
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
) -> TransformerModel:
    """Create an NLLB model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    :param cfg:
        The configuration to use.
    :param tokenizer:
        The vocabulary information to use.
    :param device:
        The device on which to initialize the model.
    """
    return NllbBuilder(cfg, vocab_info, device).build_model()


class NllbBuilder:
    """Builds modules of an NLLB model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    To tweak the model architecture, you can derive from this class and override
    the corresponding methods.
    """

    cfg: NllbConfig
    vocab_info: VocabularyInfo
    device: Optional[torch.device]

    def __init__(
        self,
        cfg: NllbConfig,
        vocab_info: VocabularyInfo,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        :param cfg:
            The configuration to use.
        :param vocab_info:
            The vocabulary information to use.
        :param device:
            The device on which to initialize modules.
        """
        self.cfg = cfg
        self.vocab_info = vocab_info
        self.device = device

    def build_model(self) -> TransformerModel:
        """Build a model."""
        embed = Embedding(
            num_embed=self.vocab_info.size,
            embed_dim=self.cfg.model_dim,
            pad_idx=self.vocab_info.pad_idx,
            scaled=True,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        frontend = self.build_frontend(embed)

        encoder = self.build_encoder()
        decoder = self.build_decoder()

        final_proj = TiedProjection(embed.weight)

        return TransformerModel(frontend, encoder, frontend, decoder, final_proj)

    def build_frontend(self, embed: Embedding) -> EncoderDecoderFrontend:
        """Build a shared encoder/decoder frontend."""
        pos_encoder = SinusoidalPositionalEncoder(
            self.cfg.model_dim,
            self.cfg.max_seq_len,
            _legacy_pad_token_idx=self.vocab_info.pad_idx,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        return TransformerTokenFrontend(
            embed,
            pos_encoder,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build an encoder."""
        layers = [self.build_encoder_layer() for _ in range(self.cfg.num_enc_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a decoder."""
        layers = [self.build_decoder_layer() for _ in range(self.cfg.num_dec_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build an encoder layer."""
        self_attn = self.build_attention(self.cfg.num_enc_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a decoder layer."""
        self_attn = self.build_attention(self.cfg.num_dec_attn_heads)

        enc_dec_attn = self.build_attention(self.cfg.num_dec_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            enc_dec_attn,
            ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a multi-head attention layer."""
        return StandardMultiheadAttention(
            num_heads,
            self.cfg.model_dim,
            attn_dropout_p=self.cfg.dropout_p,  # Applies dropout.
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a feed-forward network."""
        return StandardFeedForwardNetwork(
            self.cfg.model_dim,
            self.cfg.ffn_inner_dim,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )
