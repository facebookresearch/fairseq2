# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.wav2vec2.config import WAV2VEC2_FAMILY, Wav2Vec2Config
from fairseq2.models.wav2vec2.model import Wav2Vec2Model

get_wav2vec2_model_hub = ModelHubAccessor(
    WAV2VEC2_FAMILY, Wav2Vec2Model, Wav2Vec2Config
)
