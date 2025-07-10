# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.wav2vec2.sonar._train import (
    load_sonar_speech_trainer as load_sonar_speech_trainer,
    register_sonar_speech_train_configs as register_sonar_speech_train_configs,
    SonarSpeechTrainConfig as SonarSpeechTrainConfig,
    SonarSpeechTrainDatasetSection as SonarSpeechTrainDatasetSection,
    SonarSpeechTrainerSection as SonarSpeechTrainerSection,
    SonarSpeechTrainUnit as SonarSpeechTrainUnit,
)
