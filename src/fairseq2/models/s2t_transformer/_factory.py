# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import SiLU

from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.transformer import (
    SDPA,
    CausalAttentionBias,
    FeedForwardNetwork,
    IdentityBias,
    MultiheadAttention,
    RelativePositionalEncoding,
    RelativePositionSDPA,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEmbeddingFrontend,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerFrontend,
    TransformerModel,
    TransformerNormOrder,
    create_default_sdpa,
    init_transformer_final_projection,
)
from fairseq2.nn import (
    Embedding,
    Linear,
    PositionEncoder,
    Projection,
    SinusoidalPositionEncoder,
    StandardEmbedding,
    init_scaled_embedding,
)
from fairseq2.utils.lazy import Lazy

# isort: split

from fairseq2.models.s2t_transformer._config import S2TTransformerConfig
from fairseq2.models.s2t_transformer._feature_extractor import Conv1dFbankSubsampler
from fairseq2.models.s2t_transformer._frontend import S2TTransformerFrontend


def create_s2t_transformer_model(config: S2TTransformerConfig) -> TransformerModel:
    return S2TTransformerFactory(config).create_model()


class S2TTransformerFactory:
    _config: S2TTransformerConfig

    def __init__(self, config: S2TTransformerConfig) -> None:
        self._config = config

    def create_model(self) -> TransformerModel:
        config = self._config

        encoder_frontend = self.create_encoder_frontend()

        if config.use_conformer:
            encoder = self.create_conformer_encoder()
        else:
            encoder = self.create_encoder()

        decoder_frontend = self.create_decoder_frontend()

        decoder = self.create_decoder()

        final_proj = self.create_final_projection()

        return TransformerModel(
            encoder_frontend,
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            pad_idx=config.pad_idx,
            max_source_seq_len=config.max_source_seq_len,
            max_target_seq_len=config.max_target_seq_len,
        )

    def create_encoder_frontend(self) -> TransformerFrontend:
        config = self._config

        feat_extractor = Conv1dFbankSubsampler(
            num_channels=config.num_fbank_channels,
            inner_dim=1024,
            feature_dim=config.model_dim,
            kernel_sizes=[5, 5],
        )

        if config.use_relative_pos:
            pos_encoder = None
        else:
            pos_encoder = self.create_source_position_encoder()

        return S2TTransformerFrontend(
            config.model_dim,
            feat_extractor,
            pos_encoder,
            proj=config.use_conformer,
            dropout_p=config.dropout_p,
        )

    def create_source_position_encoder(self) -> PositionEncoder:
        config = self._config

        return SinusoidalPositionEncoder(config.model_dim, config.max_source_seq_len)

    def create_encoder(self) -> TransformerEncoder:
        config = self._config

        lazy_rel_pos_encoding = Lazy(self.create_rel_pos_encoding)

        layers = []

        for _ in range(config.num_encoder_layers):
            layer = self.create_encoder_layer(lazy_rel_pos_encoding)

            layers.append(layer)

        return StandardTransformerEncoder(layers, norm_order=TransformerNormOrder.PRE)

    def create_rel_pos_encoding(self) -> RelativePositionalEncoding:
        config = self._config

        return RelativePositionalEncoding(config.model_dim, config.max_source_seq_len)

    def create_encoder_layer(
        self, lazy_rel_pos_encoding: Lazy[RelativePositionalEncoding]
    ) -> TransformerEncoderLayer:
        self_attn = self.create_encoder_self_attention(lazy_rel_pos_encoding)

        ffn = self.create_ffn()

        return StandardTransformerEncoderLayer(
            self_attn, ffn, norm_order=TransformerNormOrder.PRE
        )

    def create_encoder_self_attention(
        self, lazy_rel_pos_encoding: Lazy[RelativePositionalEncoding]
    ) -> MultiheadAttention:
        config = self._config

        attn_bias = IdentityBias()

        sdpa: SDPA

        if config.use_relative_pos:
            rel_pos_encoding = lazy_rel_pos_encoding.retrieve()

            sdpa = RelativePositionSDPA(
                config.model_dim,
                config.num_encoder_attn_heads,
                rel_pos_encoding,
                attn_bias,
            )
        else:
            sdpa = create_default_sdpa(attn_bias, dropout_p=config.dropout_p)

        return StandardMultiheadAttention(
            config.model_dim, config.num_encoder_attn_heads, sdpa=sdpa
        )

    def create_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        config = self._config

        return StandardFeedForwardNetwork(
            config.model_dim,
            config.ffn_inner_dim,
            bias=True,
            inner_activation=SiLU() if use_swish else None,
            inner_dropout_p=config.dropout_p,
        )

    def create_conformer_encoder(self) -> TransformerEncoder:
        config = self._config

        lazy_rel_pos_encoding = Lazy(self.create_rel_pos_encoding)

        layers = []

        for _ in range(config.num_encoder_layers):
            layer = self.create_conformer_block(lazy_rel_pos_encoding)

            layers.append(layer)

        return StandardTransformerEncoder(layers, norm_order=TransformerNormOrder.POST)

    def create_conformer_block(
        self, lazy_rel_pos_encoding: Lazy[RelativePositionalEncoding]
    ) -> ConformerBlock:
        config = self._config

        ffn1 = self.create_ffn(use_swish=True)

        self_attn = self.create_encoder_self_attention(lazy_rel_pos_encoding)

        conv = self.create_conformer_conv()

        ffn2 = self.create_ffn(use_swish=True)

        return ConformerBlock(ffn1, self_attn, conv, ffn2, dropout_p=config.dropout_p)

    def create_conformer_conv(self) -> ConformerConvolution:
        config = self._config

        return ConformerConvolution(config.model_dim, config.depthwise_conv_kernel_size)

    def create_decoder_frontend(self) -> TransformerFrontend:
        config = self._config

        embed = self.create_target_embedding()

        pos_encoder = self.create_target_position_encoder()

        return TransformerEmbeddingFrontend(
            embed, pos_encoder, dropout_p=config.dropout_p
        )

    def create_target_embedding(self) -> Embedding:
        config = self._config

        return StandardEmbedding(
            num_embeddings=config.target_vocab_size,
            embedding_dim=config.model_dim,
            pad_idx=config.pad_idx,
            init_fn=init_scaled_embedding,
        )

    def create_target_position_encoder(self) -> PositionEncoder:
        config = self._config

        return SinusoidalPositionEncoder(
            config.model_dim, config.max_target_seq_len, _legacy_pad_idx=1
        )

    def create_decoder(self) -> TransformerDecoder:
        config = self._config

        layers = []

        for _ in range(config.num_decoder_layers):
            layer = self.create_decoder_layer()

            layers.append(layer)

        return StandardTransformerDecoder(layers, norm_order=TransformerNormOrder.PRE)

    def create_decoder_layer(self) -> TransformerDecoderLayer:
        config = self._config

        self_attn = self.create_decoder_self_attention()

        encoder_decoder_attn = self.create_encoder_decoder_attention()

        ffn = self.create_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
        )

    def create_decoder_self_attention(self) -> MultiheadAttention:
        config = self._config

        attn_bias = CausalAttentionBias()

        sdpa = create_default_sdpa(attn_bias, dropout_p=config.dropout_p)

        return StandardMultiheadAttention(
            config.model_dim, config.num_decoder_attn_heads, sdpa=sdpa
        )

    def create_encoder_decoder_attention(self) -> MultiheadAttention:
        config = self._config

        attn_bias = IdentityBias()

        sdpa = create_default_sdpa(attn_bias, dropout_p=config.dropout_p)

        return StandardMultiheadAttention(
            config.model_dim, config.num_decoder_attn_heads, sdpa=sdpa
        )

    def create_final_projection(self) -> Projection:
        config = self._config

        return Linear(
            config.model_dim,
            config.target_vocab_size,
            bias=False,
            init_fn=init_transformer_final_projection,
        )
