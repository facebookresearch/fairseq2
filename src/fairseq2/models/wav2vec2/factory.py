# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.nn import GELU, SiLU

from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.transformer import (
    SDPA,
    FeedForwardNetwork,
    IdentityBias,
    MultiheadAttention,
    RelativePositionalEncoding,
    RelativePositionSDPA,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.models.wav2vec2.config import Wav2Vec2Config, Wav2Vec2EncoderConfig
from fairseq2.models.wav2vec2.feature_extractor import (
    Wav2Vec2FbankFeatureExtractor,
    Wav2Vec2FeatureExtractor,
)
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend
from fairseq2.models.wav2vec2.masker import StandardWav2Vec2Masker, Wav2Vec2Masker
from fairseq2.models.wav2vec2.model import Wav2Vec2Model
from fairseq2.models.wav2vec2.position_encoder import (
    Wav2Vec2PositionEncoder,
    Wav2Vec2StackedPositionEncoder,
)
from fairseq2.models.wav2vec2.vector_quantizer import (
    GumbelWav2Vec2VectorQuantizer,
    Wav2Vec2VectorQuantizer,
)
from fairseq2.nn import (
    LayerNorm,
    PositionEncoder,
    RotaryEncoder,
    StandardLayerNorm,
    init_bert_projection,
)
from fairseq2.runtime.lazy import Lazy


def create_wav2vec2_model(config: Wav2Vec2Config) -> Wav2Vec2Model:
    return Wav2Vec2Factory(config).create_model()


class Wav2Vec2Factory:
    def __init__(self, config: Wav2Vec2Config) -> None:
        self._config = config

    def create_model(self) -> Wav2Vec2Model:
        config = self._config

        encoder_frontend = self.create_encoder_frontend()

        encoder = self.create_encoder()

        masker = self.create_masker()

        quantizer = self.create_quantizer()

        return Wav2Vec2Model(
            config.encoder_config.model_dim,
            encoder_frontend,
            encoder,
            masker,
            quantizer,
            config.final_dim,
            final_proj_bias=config.final_proj_bias,
            num_distractors=config.num_distractors,
            logit_temp=config.logit_temp,
            quantizer_encoder_grad=config.quantizer_encoder_grad,
        )

    def create_encoder_frontend(self) -> Wav2Vec2Frontend:
        config = self._config

        factory = Wav2Vec2EncoderFactory(config.encoder_config)

        return factory.create_encoder_frontend()

    def create_encoder(self) -> TransformerEncoder:
        config = self._config

        factory = Wav2Vec2EncoderFactory(config.encoder_config)

        return factory.create_encoder()

    def create_masker(self) -> Wav2Vec2Masker:
        config = self._config

        return StandardWav2Vec2Masker(
            config.encoder_config.model_dim,
            config.temporal_mask_span_len,
            config.max_temporal_mask_prob,
            config.min_num_temporal_mask_spans,
            config.spatial_mask_span_len,
            config.max_spatial_mask_prob,
            config.min_num_spatial_mask_spans,
        )

    def create_quantizer(self) -> Wav2Vec2VectorQuantizer:
        config = self._config

        return GumbelWav2Vec2VectorQuantizer(
            config.encoder_config.feature_dim,
            config.quantized_dim,
            config.num_codebooks,
            config.num_codebook_entries,
            codebook_sampling_temperature=config.codebook_sampling_temperature,
        )


class Wav2Vec2EncoderFactory:
    def __init__(self, config: Wav2Vec2EncoderConfig) -> None:
        self._config = config

    def create_encoder_frontend(self) -> Wav2Vec2Frontend:
        config = self._config

        feature_extractor = self.create_feature_extractor()

        if config.pos_encoder_type == "conv":
            pos_encoder = self.create_position_encoder()
        else:
            pos_encoder = None

        return Wav2Vec2Frontend(
            config.model_dim,
            config.feature_dim,
            feature_extractor,
            pos_encoder,
            first_pass_dropout_p=config.first_pass_dropout_p,
            layer_norm=config.layer_norm_features,
            dropout_p=config.dropout_p,
        )

    def create_feature_extractor(self) -> SequenceFeatureExtractor:
        config = self._config

        if config.use_fbank:
            return Wav2Vec2FbankFeatureExtractor(
                config.num_fbank_channels,
                config.fbank_stride,
                sample_every_k=config.sample_fbank_every_k,
            )
        else:
            return Wav2Vec2FeatureExtractor(
                config.feature_extractor_layer_descs,
                config.feature_extractor_bias,
                layer_norm=config.feature_extractor_layer_norm_convs,
                grad_scale=config.feature_grad_scale,
            )

    def create_position_encoder(self) -> PositionEncoder:
        config = self._config

        if config.pos_encoder_depth == 1:
            return Wav2Vec2PositionEncoder(
                config.model_dim,
                config.pos_conv_kernel_size,
                config.num_pos_conv_groups,
            )
        else:
            return Wav2Vec2StackedPositionEncoder(
                config.model_dim,
                config.pos_conv_kernel_size,
                config.num_pos_conv_groups,
                config.pos_encoder_depth,
            )

    def create_encoder(self) -> TransformerEncoder:
        config = self._config

        if config.use_conformer:
            return self.create_conformer_encoder()

        return self.create_transformer_encoder()

    def create_transformer_encoder(self) -> TransformerEncoder:
        config = self._config

        lazy_rel_pos_encoding = Lazy(self.create_rel_pos_encoding)

        layers = []

        for _ in range(config.num_encoder_layers):
            layer = self.create_encoder_layer(lazy_rel_pos_encoding)

            layers.append(layer)

        if config.norm_order == TransformerNormOrder.PRE:
            layer_norm = self.create_layer_norm()
        else:
            layer_norm = None

        return StandardTransformerEncoder(
            layers, layer_norm, layer_drop_p=config.layer_drop_p
        )

    def create_rel_pos_encoding(self) -> RelativePositionalEncoding:
        config = self._config

        return RelativePositionalEncoding(config.model_dim, config.max_seq_len)

    def create_encoder_layer(
        self, lazy_rel_pos_encoding: Lazy[RelativePositionalEncoding]
    ) -> TransformerEncoderLayer:
        config = self._config

        self_attn = self.create_self_attention(lazy_rel_pos_encoding)

        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn()

        ffn_layer_norm = self.create_layer_norm()

        return StandardTransformerEncoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=config.norm_order,
            dropout_p=config.dropout_p,
        )

    def create_self_attention(
        self, lazy_rel_pos_encoding: Lazy[RelativePositionalEncoding]
    ) -> MultiheadAttention:
        config = self._config

        if config.pos_encoder_type == "rotary":
            pos_encoder = RotaryEncoder(
                config.model_dim // config.num_encoder_attn_heads, config.max_seq_len
            )
        else:
            pos_encoder = None

        attn_bias = IdentityBias()

        sdpa: SDPA

        if config.pos_encoder_type == "relative":
            rel_pos_encoding = lazy_rel_pos_encoding.get()

            sdpa = RelativePositionSDPA(
                config.model_dim,
                config.num_encoder_attn_heads,
                rel_pos_encoding,
                attn_bias,
            )
        else:
            sdpa = create_default_sdpa(attn_bias, dropout_p=config.attn_dropout_p)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_encoder_attn_heads,
            sdpa,
            qkv_proj_init_fn=init_bert_projection,
            pos_encoder=pos_encoder,
            output_proj_init_fn=init_bert_projection,
        )

    def create_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        config = self._config

        activation = SiLU() if use_swish else GELU()

        return StandardFeedForwardNetwork(
            config.model_dim,
            config.ffn_inner_dim,
            bias=True,
            inner_activation=activation,
            inner_dropout_p=config.ffn_inner_dropout_p,
            proj_init_fn=init_bert_projection,
        )

    def create_conformer_encoder(self) -> TransformerEncoder:
        config = self._config

        if config.norm_order != TransformerNormOrder.POST:
            raise ValueError(
                f"`config.norm_order` must be `POST` when `config.use_conformer` is `True`, but is `{config.norm_order}` instead."
            )

        lazy_rel_pos_encoding = Lazy(self.create_rel_pos_encoding)

        layers = []

        for _ in range(config.num_encoder_layers):
            layer = self.create_conformer_block(lazy_rel_pos_encoding)

            layers.append(layer)

        return StandardTransformerEncoder(layers)

    def create_conformer_block(
        self, lazy_rel_pos_encoding: Lazy[RelativePositionalEncoding]
    ) -> ConformerBlock:
        config = self._config

        ffn1_layer_norm = self.create_layer_norm()

        ffn1 = self.create_ffn(use_swish=True)

        self_attn_layer_norm = self.create_layer_norm()

        self_attn = self.create_self_attention(lazy_rel_pos_encoding)

        conv_layer_norm = self.create_layer_norm()

        conv = self.create_conformer_conv()

        ffn2_layer_norm = self.create_layer_norm()

        ffn2 = self.create_ffn(use_swish=True)

        layer_norm = self.create_layer_norm()

        return ConformerBlock(
            ffn1_layer_norm,
            ffn1,
            self_attn_layer_norm,
            self_attn,
            conv_layer_norm,
            conv,
            ffn2_layer_norm,
            ffn2,
            layer_norm,
            dropout_p=config.dropout_p,
        )

    def create_conformer_conv(self) -> ConformerConvolution:
        config = self._config

        return ConformerConvolution(config.model_dim, config.depthwise_conv_kernel_size)

    def create_layer_norm(self) -> LayerNorm:
        config = self._config

        return StandardLayerNorm(config.model_dim, bias=True)
