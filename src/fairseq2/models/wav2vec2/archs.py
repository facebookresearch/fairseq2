# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.wav2vec2.factory import Wav2Vec2Config, Wav2Vec2EncoderConfig
from fairseq2.nn.transformer import TransformerNormOrder

wav2vec2_archs = ConfigRegistry[Wav2Vec2Config]()

wav2vec2_arch = wav2vec2_archs.decorator


@wav2vec2_arch("base")
def _base() -> Wav2Vec2Config:
    return Wav2Vec2Config()


@wav2vec2_arch("large")
def _large() -> Wav2Vec2Config:
    config = _base()

    config.encoder_config.model_dim = 1024
    config.encoder_config.num_encoder_layers = 24
    config.encoder_config.num_encoder_attn_heads = 16
    config.encoder_config.ffn_inner_dim = 4096
    config.encoder_config.dropout_p = 0.0
    config.encoder_config.layer_drop_p = 0.2
    config.quantized_dim = 768
    config.final_dim = 768

    return config


@wav2vec2_arch("large_lv60k")  # LibriVox 60k
def _large_lv60k() -> Wav2Vec2Config:
    config = _large()

    config.encoder_config.layer_norm_features = False
    config.encoder_config.feature_extractor_bias = True
    config.encoder_config.feature_extractor_layer_norm_convs = True
    config.encoder_config.layer_drop_p = 0.0
    config.encoder_config.norm_order = TransformerNormOrder.PRE
    config.codebook_sampling_temperature = (2.0, 0.1, 0.999995)

    return config


@wav2vec2_arch("pseudo_dinosr_base")
def _pseudo_dinosr_base() -> Wav2Vec2Config:
    return Wav2Vec2Config()


wav2vec2_encoder_archs = ConfigRegistry[Wav2Vec2EncoderConfig]()

wav2vec2_encoder_arch = wav2vec2_encoder_archs.decorator


@wav2vec2_encoder_arch("base")
def _base_encoder() -> Wav2Vec2EncoderConfig:
    config = _base()

    return config.encoder_config


@wav2vec2_encoder_arch("large")
def _large_encoder() -> Wav2Vec2EncoderConfig:
    config = _large()

    return config.encoder_config


@wav2vec2_encoder_arch("large_lv60k")  # LibriVox 60k
def _large_lv60k_encoder() -> Wav2Vec2EncoderConfig:
    config = _large_lv60k()

    return config.encoder_config
