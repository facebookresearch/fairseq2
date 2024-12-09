# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.w2vbert.factory import W2VBertConfig, w2vbert_arch
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig, wav2vec2_encoder_arch


@w2vbert_arch("600m")
def _600m() -> W2VBertConfig:
    return W2VBertConfig()


@w2vbert_arch("300m")
def _300m() -> W2VBertConfig:
    config = _600m()

    config.w2v2_config.encoder_config.num_encoder_layers = 12

    config.num_bert_encoder_layers = 8

    return config


@wav2vec2_encoder_arch("bert_600m")
def _600m_encoder() -> Wav2Vec2EncoderConfig:
    config = _600m()

    return config.w2v2_config.encoder_config


@wav2vec2_encoder_arch("bert_300m")
def _300m_encoder() -> Wav2Vec2EncoderConfig:
    config = _300m()

    return config.w2v2_config.encoder_config
