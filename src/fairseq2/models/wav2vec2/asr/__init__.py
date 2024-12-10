# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2.asr.factory import (
    WAV2VEC2_ASR_FAMILY as WAV2VEC2_ASR_FAMILY,
)
from fairseq2.models.wav2vec2.asr.factory import (
    Wav2Vec2AsrBuilder as Wav2Vec2AsrBuilder,
)
from fairseq2.models.wav2vec2.asr.factory import Wav2Vec2AsrConfig as Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.asr.factory import (
    create_wav2vec2_asr_model as create_wav2vec2_asr_model,
)
from fairseq2.models.wav2vec2.asr.factory import wav2vec2_asr_arch as wav2vec2_asr_arch
from fairseq2.models.wav2vec2.asr.factory import (
    wav2vec2_asr_archs as wav2vec2_asr_archs,
)
from fairseq2.models.wav2vec2.asr.loader import (
    load_wav2vec2_asr_config as load_wav2vec2_asr_config,
)
from fairseq2.models.wav2vec2.asr.loader import (
    load_wav2vec2_asr_model as load_wav2vec2_asr_model,
)
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel as Wav2Vec2AsrModel
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrOutput as Wav2Vec2AsrOutput

# isort: split

import fairseq2.models.wav2vec2.asr.archs  # Register architectures.
