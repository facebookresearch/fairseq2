# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor

# isort: split

from fairseq2.models.wav2vec2._config import Wav2Vec2Config
from fairseq2.models.wav2vec2._model import Wav2Vec2Model

get_wav2vec2_model_hub = ModelHubAccessor(Wav2Vec2Model, Wav2Vec2Config)
