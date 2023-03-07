# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, final

import torch

from fairseq2.models.s2t_transformer.arch import (
    S2TTransformerModel,
    TransformerFbankFrontend,
)
from fairseq2.models.s2t_transformer.subsampler import Conv1dFbankSubsampler
from fairseq2.models.transformer.arch import ScoreProjection, TransformerTokenFrontend
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.positional_embedding import (
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
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
class S2TTransformerConfig:
    """The default arguments correspond to the *medium* speech-to-text
    Transformer model as described in Table 3 of
    :cite:t:`https://doi.org/10.48550/arxiv.2010.05171`."""

    num_fbank_channels: int = 80
    """The number of source log-mel filterbank channels."""

    # TODO: MOVE THIS TO DICTIONARY INSTANCE!
    tgt_num_tokens: int = 100
    """The number of target tokens, e.g. vocabulary size."""

    tgt_pad_token_idx: Optional[int] = 1
    """If not ``None``, entries at ``tgt_pad_token_idx`` in target sequences
    won't contribute to the gradient."""

    max_src_len: int = 1024
    """The expected maximum source sequence length."""

    max_tgt_len: int = 1024
    """The expected maximum target sequence length."""

    model_dim: int = 512
    """The dimensionality of the model (i.e. inputs and outputs)."""

    num_enc_layers: int = 12
    """The number of encoder layers."""

    num_dec_layers: int = 6
    """The number of decoder layers."""

    num_enc_attn_heads: int = 8
    """The number of attention heads in encoder layers."""

    num_dec_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 512 * 4
    """The dimensionality of inner layers in feed-forward networks."""

    dropout_p: float = 0.15
    """The dropout probability on outputs of embedding dictionaries, attention
    layers, and feed-forward networks."""

    dtype: torch.dtype = torch.float32
    """The data type of model parameters and buffers."""


@final
class _S2TTransformerModelBuilder:
    cfg: S2TTransformerConfig
    device: Optional[torch.device]

    def __init__(self, cfg: S2TTransformerConfig) -> None:
        self.cfg = cfg

    def build(self, device: Optional[torch.device] = None) -> S2TTransformerModel:
        self.device = device

        enc_frontend = self._build_encoder_frontend()
        dec_frontend = self._build_decoder_frontend()

        enc = self._build_encoder()
        dec = self._build_decoder()

        score_proj = ScoreProjection(
            num_embed=self.cfg.tgt_num_tokens,
            embed_dim=self.cfg.model_dim,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        return S2TTransformerModel(enc_frontend, enc, dec_frontend, dec, score_proj)

    def _build_encoder_frontend(self) -> TransformerFbankFrontend:
        subsampler = Conv1dFbankSubsampler(
            num_channels=self.cfg.num_fbank_channels,
            inner_dim=1024,
            embed_dim=self.cfg.model_dim,
            kernel_sizes=[5, 5],
            device=self.device,
            dtype=self.cfg.dtype,
        )

        pos_embed = self._build_positional_embedding(self.cfg.max_src_len)

        return TransformerFbankFrontend(
            subsampler=subsampler,
            pos_embed=pos_embed,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def _build_decoder_frontend(self) -> TransformerTokenFrontend:
        embed = Embedding(
            num_embed=self.cfg.tgt_num_tokens,
            embed_dim=self.cfg.model_dim,
            pad_idx=self.cfg.tgt_pad_token_idx,
            scaled=True,
            device=self.device,
            dtype=self.cfg.dtype,
        )

        pos_embed = self._build_positional_embedding(self.cfg.max_tgt_len)

        return TransformerTokenFrontend(
            embed=embed,
            pos_embed=pos_embed,
            dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def _build_positional_embedding(self, max_seq_len: int) -> PositionalEmbedding:
        return SinusoidalPositionalEmbedding(
            max_seq_len=max_seq_len,
            embed_dim=self.cfg.model_dim,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def _build_encoder(self) -> TransformerEncoder:
        layers = [self._build_encoder_layer() for _ in range(self.cfg.num_enc_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def _build_decoder(self) -> TransformerDecoder:
        layers = [self._build_decoder_layer() for _ in range(self.cfg.num_dec_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def _build_encoder_layer(self) -> TransformerEncoderLayer:
        self_attn = self._build_attention(self.cfg.num_enc_attn_heads)

        ffn = self._build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn=self_attn,
            ffn=ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def _build_decoder_layer(self) -> TransformerDecoderLayer:
        self_attn = self._build_attention(self.cfg.num_dec_attn_heads)

        enc_dec_attn = self._build_attention(self.cfg.num_dec_attn_heads)

        ffn = self._build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn=self_attn,
            enc_dec_attn=enc_dec_attn,
            ffn=ffn,
            dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def _build_attention(self, num_heads: int) -> MultiheadAttention:
        return StandardMultiheadAttention(
            num_heads=num_heads,
            model_dim=self.cfg.model_dim,
            attn_dropout_p=self.cfg.dropout_p,
            device=self.device,
            dtype=self.cfg.dtype,
        )

    def _build_ffn(self) -> FeedForwardNetwork:
        return StandardFeedForwardNetwork(
            model_dim=self.cfg.model_dim,
            inner_dim=self.cfg.ffn_inner_dim,
            inner_dropout_p=self.cfg.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.cfg.dtype,
        )


def create_s2t_transformer_model(
    cfg: S2TTransformerConfig, device: Optional[torch.device] = None
) -> S2TTransformerModel:
    """Build a model that follows the speech-to-text Transformer architecture
    as described in :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.

    :param cfg:
        The configuration to use.
    """
    return _S2TTransformerModelBuilder(cfg).build(device)
