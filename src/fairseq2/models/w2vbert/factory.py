# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.w2vbert.config import W2VBertConfig
from fairseq2.models.w2vbert.model import W2VBertModel
from fairseq2.models.wav2vec2 import Wav2Vec2Factory, Wav2Vec2Model


def create_w2vbert_model(config: W2VBertConfig) -> W2VBertModel:
    return W2VBertFactory(config).create_model()


class W2VBertFactory:
    def __init__(self, config: W2VBertConfig) -> None:
        self._config = config

    def create_model(self) -> W2VBertModel:
        config = self._config

        encoder_config = config.w2v2_config.encoder_config

        if encoder_config.layer_drop_p != 0.0:
            raise ValueError(
                f"`config.w2v2_config.encoder_config.layer_drop_p` must be 0.0 since w2v-BERT does not support LayerDrop, but is {encoder_config.layer_drop_p} instead."
            )

        if config.num_bert_encoder_layers >= encoder_config.num_encoder_layers:
            raise ValueError(
                f"`config.num_bert_encoder_layers` must be less than `w2v2_config.encoder_config.num_encoder_layers` ({encoder_config.num_encoder_layers}), but is {config.num_bert_encoder_layers} instead."
            )

        if config.num_target_codebooks > config.w2v2_config.num_codebooks:
            raise ValueError(
                f"`config.num_target_codebooks` must be less than `config.w2v2_config.num_codebooks` ({config.w2v2_config.num_codebooks}), but is {config.num_target_codebooks} instead."
            )

        w2v2_model = self.create_wav2vec2_model()

        return W2VBertModel(
            w2v2_model,
            config.num_bert_encoder_layers,
            num_target_codebooks=config.num_target_codebooks,
        )

    def create_wav2vec2_model(self) -> Wav2Vec2Model:
        config = self._config

        factory = Wav2Vec2Factory(config.w2v2_config)

        return factory.create_model()
