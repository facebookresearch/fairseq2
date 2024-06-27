# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.wav2vec2.archs import wav2vec2_encoder_archs
from fairseq2.models.wav2vec2.asr.factory import Wav2Vec2AsrConfig

wav2vec2_asr_archs = ConfigRegistry[Wav2Vec2AsrConfig]()

wav2vec2_asr_arch = wav2vec2_asr_archs.decorator


def _base_10h() -> Wav2Vec2AsrConfig:
    return Wav2Vec2AsrConfig()


def _base_100h() -> Wav2Vec2AsrConfig:
    config = _base_10h()

    config.encoder_config.layer_drop_p = 0.1

    return config


def _large_10h() -> Wav2Vec2AsrConfig:
    config = _base_10h()

    config.encoder_config = wav2vec2_encoder_archs.get("large")
    config.encoder_config.feature_gradient_scale = 1.0
    config.encoder_config.dropout_p = 0.0
    config.encoder_config.attn_dropout_p = 0.0
    config.encoder_config.ffn_inner_dropout_p = 0.1
    config.encoder_config.layer_drop_p = 0.1

    config.max_temporal_mask_prob = 0.80
    config.max_spatial_mask_prob = 0.30

    return config


def _large_100h() -> Wav2Vec2AsrConfig:
    config = _large_10h()

    config.max_temporal_mask_prob = 0.53
    config.max_spatial_mask_prob = 0.55

    return config


def _large_lv60k_10h() -> Wav2Vec2AsrConfig:
    config = _base_10h()

    config.encoder_config = wav2vec2_encoder_archs.get("large_lv60k")
    config.encoder_config.feature_gradient_scale = 1.0
    config.encoder_config.dropout_p = 0.0
    config.encoder_config.attn_dropout_p = 0.0
    config.encoder_config.ffn_inner_dropout_p = 0.1
    config.encoder_config.layer_drop_p = 0.1

    config.max_temporal_mask_prob = 0.80
    config.max_spatial_mask_prob = 0.30

    return config


def _large_lv60k_100h() -> Wav2Vec2AsrConfig:
    config = _large_lv60k_10h()

    config.max_temporal_mask_prob = 0.53
    config.max_spatial_mask_prob = 0.55

    return config


def _register_wav2vec2_asr_archs() -> None:
    # fmt: off
    wav2vec2_asr_archs.register("base_10h",         _base_10h)
    wav2vec2_asr_archs.register("base_100h",        _base_100h)
    wav2vec2_asr_archs.register("large_10h",        _large_10h)
    wav2vec2_asr_archs.register("large_100h",       _large_100h)
    wav2vec2_asr_archs.register("large_lv60k_10h",  _large_lv60k_10h)
    wav2vec2_asr_archs.register("large_lv60k_100h", _large_lv60k_100h)
    # fmt: on
