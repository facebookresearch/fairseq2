# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2.asr.factory import Wav2Vec2AsrConfig, wav2vec2_asr_arch
from fairseq2.models.wav2vec2.factory import wav2vec2_encoder_archs


@wav2vec2_asr_arch("base_10h")
def _base_10h() -> Wav2Vec2AsrConfig:
    return Wav2Vec2AsrConfig()

##################################################################
# Register the new wav2vec2_asr_arch here which returns a Wav2Vec2AsrConfig: 
# Here you will change adapt the wav2vec2 config for ASR finetuning.
@wav2vec2_asr_arch("mms_base_300m_asr")
def _mms_base_300m_eng_accent() -> Wav2Vec2AsrConfig:
    config = _base_10h()
    config.encoder_config = wav2vec2_encoder_archs.get("mms_base_300m")

    # ENCODER – from large_lv60k_10h preset # FIXME: Vineel
    config.encoder_config.feature_gradient_scale = 1.0
    config.encoder_config.dropout_p = 0.0
    config.encoder_config.attn_dropout_p = 0.0
    config.encoder_config.ffn_inner_dropout_p = 0.1
    config.encoder_config.layer_drop_p = 0.1
    config.max_temporal_mask_prob = 0.80
    config.max_spatial_mask_prob = 0.30

    return config
#############################################################################

@wav2vec2_asr_arch("base_100h")
def _base_100h() -> Wav2Vec2AsrConfig:
    config = _base_10h()

    config.encoder_config.layer_drop_p = 0.1

    return config


@wav2vec2_asr_arch("large_10h")
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


@wav2vec2_asr_arch("large_100h")
def _large_100h() -> Wav2Vec2AsrConfig:
    config = _large_10h()

    config.max_temporal_mask_prob = 0.53
    config.max_spatial_mask_prob = 0.55

    return config


@wav2vec2_asr_arch("large_lv60k_10h")
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


@wav2vec2_asr_arch("large_lv60k_100h")
def _large_lv60k_100h() -> Wav2Vec2AsrConfig:
    config = _large_lv60k_10h()

    config.max_temporal_mask_prob = 0.53
    config.max_spatial_mask_prob = 0.55

    return config