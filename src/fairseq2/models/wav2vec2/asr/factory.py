# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch.nn as nn

from fairseq2.models.transformer import TransformerEncoder
from fairseq2.models.wav2vec2 import (
    StandardWav2Vec2Masker,
    Wav2Vec2EncoderFactory,
    Wav2Vec2Frontend,
    Wav2Vec2Masker,
)
from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel
from fairseq2.nn import Linear, Projection


def create_wav2vec2_asr_model(config: Wav2Vec2AsrConfig) -> Wav2Vec2AsrModel:
    return Wav2Vec2AsrFactory(config).create_model()


class Wav2Vec2AsrFactory:
    def __init__(self, config: Wav2Vec2AsrConfig) -> None:
        self._config = config

    def create_model(self) -> Wav2Vec2AsrModel:
        config = self._config

        encoder_frontend = self.create_encoder_frontend()

        encoder = self.create_encoder()

        if config.use_masking:
            masker = self.create_masker()
        else:
            masker = None

        final_proj = self.create_final_projection()

        return Wav2Vec2AsrModel(
            config.encoder_config.model_dim,
            encoder_frontend,
            encoder,
            final_proj,
            masker=masker,
            final_dropout_p=config.final_dropout_p,
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

    def create_final_projection(self) -> Projection:
        config = self._config

        return Linear(
            config.encoder_config.model_dim,
            config.target_vocab_size,
            bias=True,
            init_fn=_init_final_projection,
        )


def _init_final_projection(proj: Linear) -> None:
    nn.init.xavier_uniform_(proj.weight)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)
