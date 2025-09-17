# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from fairseq2.models.transformer import (
    create_default_sdpa,
    create_standard_layer_norm,
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEmbeddingFrontend,
    TransformerEncoder,
    TransformerFrontend,
)

from fairseq2.models.wav2vec2 import (
    StandardWav2Vec2Masker,
    Wav2Vec2EncoderConfig,
    Wav2Vec2EncoderFactory,
    Wav2Vec2Frontend,
    Wav2Vec2Masker,
)
from fairseq2.models.wav2vec2.sonar._config import SonarSpeechEncoderConfig
from fairseq2.models.wav2vec2.sonar._model import SonarSpeechEncoderModel
from fairseq2.models.wav2vec2.sonar._pooler import (
    AttentionEncoderOutputPooler,
    EncoderOutputPooler,
    MeanEncoderOutputPooler,
    SelfAttentiveEncoderOutputPooler,
)
from fairseq2.nn import (
    Embedding,
    init_scaled_embedding,
    LayerNorm,
    Linear,
    PositionEncoder,
    SinusoidalPositionEncoder,
    StandardEmbedding,
)


def create_sonar_speech_model(
    config: SonarSpeechEncoderConfig,
) -> SonarSpeechEncoderModel:
    return SonarSpeechEncoderFactory(config).create_model()


class SonarSpeechEncoderFactory:
    config: SonarSpeechEncoderConfig

    def __init__(self, config: SonarSpeechEncoderConfig) -> None:
        if config.encoder_config.model_dim != config.model_dim:
            raise ValueError(
                f"`config.model_dim` and `config.encoder_config.model_dim` must be equal, but are {config.model_dim} and {config.encoder_config.model_dim} instead."
            )

        self.config = config

    def create_model(self) -> SonarSpeechEncoderModel:
        encoder_frontend, encoder = self.create_encoder()

        if self.config.use_masking:
            masker = self.create_masker()
        else:
            masker = None

        return SonarSpeechEncoderModel(
            encoder_frontend=encoder_frontend,
            encoder=encoder,
            layer_norm=self.create_w2v2_final_layer_norm(),
            final_dropout_p=self.config.final_dropout_p,
            # encoder_pooler=self.create_attention_pooler(),
            encoder_pooler=self.create_pooler(),
            masker=masker,
        )

    def create_encoder(self) -> tuple[Wav2Vec2Frontend, TransformerEncoder]:
        factory = Wav2Vec2EncoderFactory(self.config.encoder_config)

        encoder_frontend = factory.create_encoder_frontend()

        encoder = factory.create_encoder()

        return encoder_frontend, encoder

    def create_masker(self) -> Wav2Vec2Masker:
        config = self.config

        return StandardWav2Vec2Masker(
            config.mask_codebase,
            config.encoder_config.model_dim,
            config.temporal_mask_span_len,
            config.max_temporal_mask_prob,
            config.min_num_temporal_mask_spans,
            config.spatial_mask_span_len,
            config.max_spatial_mask_prob,
            config.min_num_spatial_mask_spans,
        )

    def create_attention_pooler(self) -> EncoderOutputPooler:
        return AttentionEncoderOutputPooler(
            decoder_frontend=self.create_decoder_frontend(),
            decoder=self.create_decoder(),
            projection_out=self.create_projection_out(),
            bos_idx=self.config.bos_idx,
        )

    def create_mean_pooler(self) -> EncoderOutputPooler:
        return MeanEncoderOutputPooler(projection_out=self.create_projection_out())

    def create_sa_pooler(self) -> EncoderOutputPooler:
        return SelfAttentiveEncoderOutputPooler(
            h_linear=self.create_h_linear(),
            attention=self.create_attn_weight(),
            projection_out=self.create_projection_out(),
        )

    def create_pooler(self):
        if self.config.pooling_type == "attention":
            return self.create_attention_pooler()
        elif self.config.pooling_type == "mean":
            return self.create_mean_pooler()
        elif self.config.pooling_type == "sa":
            return self.create_sa_pooler()
        else:
            raise ValueError(f"Unknown pooling type: {self.config.pooling_type}")

    def create_decoder_frontend(self) -> TransformerFrontend:
        return TransformerEmbeddingFrontend(
            self.create_embedding(),
            self.create_pos_encoder(),
            dropout_p=self.config.dropout_p,
        )

    def create_pos_encoder(self) -> PositionEncoder:
        return SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.max_seq_len,
        )

    def create_embedding(self) -> Embedding:
        return StandardEmbedding(
            num_embeddings=self.config.encoder_config.model_dim,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.pad_idx,
            init_fn=init_scaled_embedding,
        )

    def create_decoder(self) -> TransformerDecoder:
        num_layers = self.config.num_decoder_layers
        layers = [self.create_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            # norm_order=self.config.decoder_norm_order,
        )

    def create_decoder_layer(self) -> TransformerDecoderLayer:
        num_heads = self.config.num_decoder_attn_heads

        return StandardTransformerDecoderLayer(
            self.create_attention(num_heads),
            self.create_attention(num_heads),
            self.create_ffn(),
            dropout_p=self.config.dropout_p,
            # norm_order=self.config.decoder_norm_order,
        )

    def create_attention(self, num_heads: int) -> MultiheadAttention:
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
        )

    def create_ffn(self) -> FeedForwardNetwork:
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            # norm_order=self.config.decoder_norm_order,
        )

    def create_w2v2_final_layer_norm(self) -> Optional[LayerNorm]:
        if not self.config.encoder_config.use_conformer:
            return None

        return create_standard_layer_norm(
            self.config.encoder_config.model_dim,
        )

    def create_projection_out(self) -> Linear:
        proj = Linear(
            input_dim=self.config.model_dim,
            output_dim=self.config.embedd_dim,
            bias=False,
        )
        torch.nn.init.normal_(proj.weight, mean=0, std=1e-3)

        return proj

    def create_h_linear(self) -> Linear:
        proj = Linear(
            input_dim=self.config.model_dim,
            output_dim=self.config.model_dim,
            bias=True,
        )

        return proj

    def create_attn_weight(self) -> Linear:
        weight_mat = Linear(
            input_dim=self.config.model_dim,
            output_dim=1,
            bias=False,
        )

        return weight_mat
