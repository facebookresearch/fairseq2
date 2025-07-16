# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor

# isort: split

from fairseq2.models.wav2vec2.asr._config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.asr._model import Wav2Vec2AsrModel

get_wav2vec2_asr_model_hub = ModelHubAccessor(Wav2Vec2AsrModel, Wav2Vec2AsrConfig)
