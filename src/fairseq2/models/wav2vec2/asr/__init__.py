# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2.asr._factory import (
    init_final_projection as init_final_projection,
)
from fairseq2.models.wav2vec2.asr._model import Wav2Vec2AsrModel as Wav2Vec2AsrModel
from fairseq2.models.wav2vec2.asr.checkpoint import (
    ModelHubAccessor,
)
from fairseq2.models.wav2vec2.asr.checkpoint import (
    _convert_wav2vec2_asr_checkpoint as _convert_wav2vec2_asr_checkpoint,  # isort: split
)
from fairseq2.models.wav2vec2.asr.checkpoint import (
    convert_wav2vec2_asr_checkpoint as convert_wav2vec2_asr_checkpoint,
)
from fairseq2.models.wav2vec2.asr.checkpoint import (
    fairseq2.models,
    fairseq2.models.wav2vec2.asr._checkpoint,
    from,
    import,
)
from fairseq2.models.wav2vec2.asr.config import (
    WAV2VEC2_ASR_FAMILY as WAV2VEC2_ASR_FAMILY,
)
from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig as Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.asr.config import (
    _register_wav2vec2_asr_configs as _register_wav2vec2_asr_configs,
)
from fairseq2.models.wav2vec2.asr.factory import (
    Wav2Vec2AsrFactory as Wav2Vec2AsrFactory,
)
from fairseq2.models.wav2vec2.asr.factory import (
    _create_wav2vec2_asr_model as _create_wav2vec2_asr_model,
)
from fairseq2.models.wav2vec2.asr.hub import (
    get_wav2vec2_asr_model_hub as get_wav2vec2_asr_model_hub,
)
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel as Wav2Vec2AsrModel

get_wav2vec2_asr_model_hub = ModelHubAccessor(Wav2Vec2AsrModel, Wav2Vec2AsrConfig)
