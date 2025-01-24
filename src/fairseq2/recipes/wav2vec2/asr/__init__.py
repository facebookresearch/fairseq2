# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.wav2vec2.asr._train import (
    Wav2Vec2AsrTrainConfig as Wav2Vec2AsrTrainConfig,
)
from fairseq2.recipes.wav2vec2.asr._train import (
    Wav2Vec2AsrTrainDatasetSection as Wav2Vec2AsrTrainDatasetSection,
)
from fairseq2.recipes.wav2vec2.asr._train import (
    Wav2Vec2AsrTrainerSection as Wav2Vec2AsrTrainerSection,
)
from fairseq2.recipes.wav2vec2.asr._train import (
    Wav2Vec2AsrTrainUnit as Wav2Vec2AsrTrainUnit,
)
from fairseq2.recipes.wav2vec2.asr._train import (
    load_wav2vec2_asr_trainer as load_wav2vec2_asr_trainer,
)
from fairseq2.recipes.wav2vec2.asr._train import (
    register_wav2vec2_asr_train_configs as register_wav2vec2_asr_train_configs,
)
