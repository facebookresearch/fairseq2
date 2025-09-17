# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.context import RuntimeContext
from fairseq2.data import VocabularyInfo
from fairseq2.models.transformer import TransformerNormOrder
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrConfig

WAV2VEC2_SONAR_SPEECH_MODEL_FAMILY: Final = "wav2vec2_sonar_speech"


@dataclass
class SonarSpeechEncoderConfig:
    """Holds the configuration of a Sonar model."""

    encoder_config: Wav2Vec2EncoderConfig = field(
        default_factory=lambda: Wav2Vec2EncoderConfig(
            feature_gradient_scale=1.0,
            dropout_p=0.0,
            attn_dropout_p=0.0,
            ffn_inner_dropout_p=0.1,
        )
    )
    """The configuration of the wav2vec 2.0 encoder model."""

    final_dropout_p: float = 0.1
    """The dropout probability applied final projection"""

    model_dim: int = 1024
    """The encoder output embedding dimension."""

    embedd_dim: int = 1024
    """The target embedding dimension."""

    max_seq_len: int = 1024
    """The expected maximum sequence length."""

    pad_idx: int = 1
    """The index of the pad symbol in the vocabulary."""

    bos_idx: int = 2
    """The index of bos symbol used in attention pooling"""

    pooling_type: str = "attention"
    """The pooling type for speech embedding"""

    num_decoder_layers: int = 6
    """The number of Transformer decoder layers."""

    num_decoder_attn_heads: int = 16
    """The number of attention heads in Transformer decoder layers."""

    decoder_norm_order: TransformerNormOrder = TransformerNormOrder.POST
    """Layer norm order in decoder modules."""

    ffn_inner_dim: int = 4096
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float = 0.1
    """The dropout probability in Transformer layers."""

    # Mask
    mask_codebase: str = "fairseq2"

    use_masking: bool = True
    """If ``True``, masks features as regularization."""

    temporal_mask_span_len: int = 10
    """The length of each temporal mask span that is applied over time steps."""

    max_temporal_mask_prob: float = 0.5
    """The maximum probability of masking a time step. Note that, due to mask
    span overlap, the effective probability will be lower."""

    min_num_temporal_mask_spans: int = 2
    """The minimum number of temporal masks sampled per sequence."""

    spatial_mask_span_len: int = 64
    """The length of each spatial mask span that is applied over features."""

    max_spatial_mask_prob: float = 0.2
    """The maximum probability of masking a feature. Note that, due to mask span
    overlap, the effective probability will be lower."""

    min_num_spatial_mask_spans: int = 2
    """The minimum number of spatial masks sampled per sequence."""


def register_sonar_speech_encoder_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(SonarSpeechEncoderConfig)

    arch = registry.decorator

    w2v2_encoder_registry = context.get_config_registry(Wav2Vec2EncoderConfig)

    @arch("base_10h")
    def base_10h() -> SonarSpeechEncoderConfig:

        return SonarSpeechEncoderConfig()

    @arch("7b")
    def sonar_7b() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("7b")

        config.model_dim = 2048
        config.embedd_dim = 1024
        config.num_decoder_layers = 3

        return config

    @arch("3b")
    def sonar_3b() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("3b")

        config.model_dim = 2048
        config.embedd_dim = 1024
        config.num_decoder_layers = 3

        return config

    @arch("1b")
    def sonar_1b() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("1b")

        config.model_dim = 1280
        config.num_decoder_layers = 3

        return config

    @arch("7b_mean")
    def sonar_7b_mean() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("7b")

        config.model_dim = 2048
        config.embedd_dim = 1024
        config.pooling_type = "mean"

        return config

    @arch("1b_mean")
    def sonar_1b_mean() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("1b")

        config.model_dim = 1280
        config.pooling_type = "mean"

        return config

    @arch("7b_sa")
    def sonar_7b_sa() -> SonarSpeechEncoderConfig:
        config = sonar_7b()
        config.pooling_type = "sa"

        return config

    @arch("1b_sa")
    def sonar_1b_sa() -> SonarSpeechEncoderConfig:
        config = sonar_1b()
        config.pooling_type = "sa"

        return config

    @arch("7b_noaug")
    def sonar_7b() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("7b")

        config.model_dim = 2048
        config.embedd_dim = 1024
        config.num_decoder_layers = 3
        config.use_masking = False

        return config
