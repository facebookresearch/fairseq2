# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch

from fairseq2.data.text import VocabularyInfo
from fairseq2.models.transformer.model import TransformerModel, TransformerTokenFrontend
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.positional_embedding import SinusoidalPositionalEmbedding
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
class TransformerConfig:
    """Configuration of a Transformer model.

    The default values correspond to the *base* architecture described in
    Table 3 of :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.
    """

    max_seq_len: int = 1024
    """The expected maximum sequence length."""

    model_dim: int = 512
    """The dimensionality of the model (i.e. inputs and outputs)."""

    num_enc_layers: int = 6
    """The number of encoder layers."""

    num_dec_layers: int = 6
    """The number of decoder layers."""

    num_enc_attn_heads: int = 8
    """The number of attention heads in encoder layers."""

    num_dec_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 2048
    """The dimensionality of inner layers in feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of embedding dictionaries, attention
    layers, and feed-forward networks."""

    norm_order: TransformerNormOrder = TransformerNormOrder.POST
    """The Layer Normalization order."""

    legacy_pos_embed: bool = False
    """If ``True``, sinusoidal positional embeddings will be initialized in a
    way that is compatible with the original fairseq."""

    dtype: torch.dtype = torch.float32
    """The data type of model parameters and buffers."""


def create_transformer_model(
    cfg: TransformerConfig,
    vocab_info: VocabularyInfo,
    device: Optional[torch.device] = None,
) -> TransformerModel:
    """Create a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.

    :param cfg:
        The configuration to use.
    :param vocab_info:
        The vocabulary information to use.
    :param device:
        The device on which to initialize the model.
    """
    return TransformerBuilder(cfg, vocab_info, device).build_model()


class TransformerBuilder:
    """Builds modules of a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`.

    To tweak the model architecture, you can derive from this class and override
    the corresponding methods.
    """

    cfg: TransformerConfig
    vocab_info: VocabularyInfo
    device: Optional[torch.device]

    def __init__(
        self,
        cfg: TransformerConfig,
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
        frontend = self.build_frontend()

        enc = self.build_encoder()
        dec = self.build_decoder()

        score_proj = TiedProjection(frontend.embed.weight)

        return TransformerModel(frontend, enc, frontend, dec, score_proj)

    def build_frontend(self) -> TransformerTokenFrontend:
        """Build a shared encoder/decoder frontend."""
        embed = Embedding(
            num_embed=self.vocab_info.size,
            embed_dim=self.cfg.model_dim,
            pad_idx=self.vocab_info.pad_idx,
            scaled=True,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        if self.cfg.legacy_pos_embed:
            pad_token_idx = self.vocab_info.pad_idx
        else:
            pad_token_idx = None

        pos_embed = SinusoidalPositionalEmbedding(
            max_seq_len=self.cfg.max_seq_len,
            embed_dim=self.cfg.model_dim,
            legacy_pad_token_idx=pad_token_idx,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        return TransformerTokenFrontend(
            embed,
            pos_embed,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build an encoder."""
        layers = [self.build_encoder_layer() for _ in range(self.cfg.num_enc_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=self.cfg.norm_order,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a decoder."""
        layers = [self.build_decoder_layer() for _ in range(self.cfg.num_dec_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=self.cfg.norm_order,
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
            norm_order=self.cfg.norm_order,
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
            norm_order=self.cfg.norm_order,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a multi-head attention layer."""
        return StandardMultiheadAttention(
            num_heads, self.cfg.model_dim, device=self.device, dtype=self.cfg.dtype
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a feed-forward network."""
        return StandardFeedForwardNetwork(
            self.cfg.model_dim,
            self.cfg.ffn_inner_dim,
            norm_order=self.cfg.norm_order,
            device=self.device,
            dtype=self.cfg.dtype,
        )
