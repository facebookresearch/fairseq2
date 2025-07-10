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


def register_sonar_speech_encoder_configs(context: RuntimeContext) -> None:
    registry = context.get_config_registry(SonarSpeechEncoderConfig)

    arch = registry.decorator

    w2v2_encoder_registry = context.get_config_registry(Wav2Vec2EncoderConfig)

    @arch("base_10h")
    def base_10h() -> SonarSpeechEncoderConfig:

        return SonarSpeechEncoderConfig()

    @arch("7b_fleurs")
    def fleurs_7b() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("7b")

        config.model_dim = 2048
        config.embedd_dim = 1024
        config.num_decoder_layers = 6

        return config

    @arch("1b_fleurs")
    def fleurs_1b() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("1b")

        config.model_dim = 1280
        config.num_decoder_layers = 3

        return config

    @arch("7b_fleurs_mean")
    def fleurs_7b_mean() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("7b")

        config.model_dim = 2048
        config.embedd_dim = 1024
        config.pooling_type = "mean"

        return config

    @arch("1b_fleurs_mean")
    def fleurs_1b_mean() -> SonarSpeechEncoderConfig:
        config = SonarSpeechEncoderConfig()
        config.encoder_config = w2v2_encoder_registry.get("1b")

        config.model_dim = 1280
        config.pooling_type = "mean"

        return config
