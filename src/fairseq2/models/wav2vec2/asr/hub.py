# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.wav2vec2.asr.config import WAV2VEC2_ASR_FAMILY, Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel

get_wav2vec2_asr_model_hub = ModelHubAccessor(
    WAV2VEC2_ASR_FAMILY, Wav2Vec2AsrModel, Wav2Vec2AsrConfig
)
