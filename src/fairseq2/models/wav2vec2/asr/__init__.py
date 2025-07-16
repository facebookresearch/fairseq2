# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2.asr._checkpoint import (
    convert_wav2vec2_asr_checkpoint as convert_wav2vec2_asr_checkpoint,
)
from fairseq2.models.wav2vec2.asr._config import (
    WAV2VEC2_ASR_MODEL_FAMILY as WAV2VEC2_ASR_MODEL_FAMILY,
)
from fairseq2.models.wav2vec2.asr._config import Wav2Vec2AsrConfig as Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.asr._config import (
    register_wav2vec2_asr_configs as register_wav2vec2_asr_configs,
)
from fairseq2.models.wav2vec2.asr._factory import (
    Wav2Vec2AsrFactory as Wav2Vec2AsrFactory,
)
from fairseq2.models.wav2vec2.asr._factory import (
    create_wav2vec2_asr_model as create_wav2vec2_asr_model,
)
from fairseq2.models.wav2vec2.asr._hub import (
    get_wav2vec2_asr_model_hub as get_wav2vec2_asr_model_hub,
)
from fairseq2.models.wav2vec2.asr._model import Wav2Vec2AsrModel as Wav2Vec2AsrModel
