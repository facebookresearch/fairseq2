# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2 import (
    StandardWav2Vec2Masker,
    Wav2Vec2EncoderFactory,
    Wav2Vec2Frontend,
    Wav2Vec2Masker,
)
from fairseq2.models.wav2vec2.asr._config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.asr._model import Wav2Vec2AsrModel
from fairseq2.nn.transformer import TransformerEncoder


class Wav2Vec2AsrFactory:
    _config: Wav2Vec2AsrConfig

    def __init__(self, config: Wav2Vec2AsrConfig) -> None:
        self._config = config

    def create_model(self) -> Wav2Vec2AsrModel:
        config = self._config

        encoder_frontend, encoder = self.create_encoder()

        if config.use_masking:
            masker = self.create_masker()
        else:
            masker = None

        return Wav2Vec2AsrModel(
            encoder_frontend,
            encoder,
            config.vocab_info,
            masker=masker,
            final_dropout_p=config.final_dropout_p,
        )

    def create_encoder(self) -> tuple[Wav2Vec2Frontend, TransformerEncoder]:
        config = self._config

        factory = Wav2Vec2EncoderFactory(config.encoder_config)

        encoder_frontend = factory.create_encoder_frontend()

        encoder = factory.create_encoder()

        return encoder_frontend, encoder

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
