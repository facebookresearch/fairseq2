# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# isort: split

from fairseq2.models import ModelHubAccessor

from fairseq2.models.wav2vec2.sonar._config import (
    register_sonar_speech_encoder_configs as register_sonar_speech_encoder_configs,
    SonarSpeechEncoderConfig as SonarSpeechEncoderConfig,
    WAV2VEC2_SONAR_SPEECH_MODEL_FAMILY as WAV2VEC2_SONAR_SPEECH_MODEL_FAMILY,
)
from fairseq2.models.wav2vec2.sonar._factory import (
    create_sonar_speech_model as create_sonar_speech_model,
    SonarSpeechEncoderFactory as SonarSpeechEncoderFactory,
)
from fairseq2.models.wav2vec2.sonar._model import (
    SonarSpeechEncoderModel as SonarSpeechEncoderModel,
)

get_wav2vec2_sonar_model_hub = ModelHubAccessor(
    SonarSpeechEncoderModel, SonarSpeechEncoderConfig
)
