# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.bestrq.factory import (
    BestRQConfig,
    bestrq_arch,
    bestrq_encoder_arch,
)
from fairseq2.models.wav2vec2.factory import (
    Wav2Vec2EncoderConfig,
)
from fairseq2.nn.transformer import TransformerNormOrder


def register_archs() -> None:
    @bestrq_arch("base")
    def _base() -> BestRQConfig:
        return BestRQConfig()

    @bestrq_encoder_arch("base")
    def _base_encoder() -> Wav2Vec2EncoderConfig:
        config = _base()

        return config.encoder_config