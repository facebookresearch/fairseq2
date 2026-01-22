# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2.ac import apply_ac_to_wav2vec2 as apply_ac_to_wav2vec2
from fairseq2.models.wav2vec2.config import WAV2VEC2_FAMILY as WAV2VEC2_FAMILY
from fairseq2.models.wav2vec2.config import Wav2Vec2Config as Wav2Vec2Config
from fairseq2.models.wav2vec2.config import (
    Wav2Vec2EncoderConfig as Wav2Vec2EncoderConfig,
)
from fairseq2.models.wav2vec2.config import (
    register_wav2vec2_configs as register_wav2vec2_configs,
)
from fairseq2.models.wav2vec2.factory import (
    Wav2Vec2EncoderFactory as Wav2Vec2EncoderFactory,
)
from fairseq2.models.wav2vec2.factory import Wav2Vec2Factory as Wav2Vec2Factory
from fairseq2.models.wav2vec2.factory import (
    create_wav2vec2_model as create_wav2vec2_model,
)
from fairseq2.models.wav2vec2.feature_extractor import (
    Wav2Vec2FbankFeatureExtractor as Wav2Vec2FbankFeatureExtractor,
)
from fairseq2.models.wav2vec2.feature_extractor import (
    Wav2Vec2FeatureExtractor as Wav2Vec2FeatureExtractor,
)
from fairseq2.models.wav2vec2.frontend import Wav2Vec2Frontend as Wav2Vec2Frontend
from fairseq2.models.wav2vec2.fsdp import (
    apply_fsdp_to_wav2vec2 as apply_fsdp_to_wav2vec2,
)
from fairseq2.models.wav2vec2.hub import (
    get_wav2vec2_model_hub as get_wav2vec2_model_hub,
)
from fairseq2.models.wav2vec2.interop import (
    convert_wav2vec2_state_dict as convert_wav2vec2_state_dict,
)
from fairseq2.models.wav2vec2.masker import (
    StandardWav2Vec2Masker as StandardWav2Vec2Masker,
)
from fairseq2.models.wav2vec2.masker import Wav2Vec2Masker as Wav2Vec2Masker
from fairseq2.models.wav2vec2.model import Wav2Vec2Features as Wav2Vec2Features
from fairseq2.models.wav2vec2.model import Wav2Vec2Loss as Wav2Vec2Loss
from fairseq2.models.wav2vec2.model import Wav2Vec2Model as Wav2Vec2Model
from fairseq2.models.wav2vec2.model import Wav2Vec2Output as Wav2Vec2Output
from fairseq2.models.wav2vec2.position_encoder import (
    Wav2Vec2PositionEncoder as Wav2Vec2PositionEncoder,
)
from fairseq2.models.wav2vec2.position_encoder import (
    Wav2Vec2StackedPositionEncoder as Wav2Vec2StackedPositionEncoder,
)
from fairseq2.models.wav2vec2.vector_quantizer import (
    GumbelWav2Vec2VectorQuantizer as GumbelWav2Vec2VectorQuantizer,
)
from fairseq2.models.wav2vec2.vector_quantizer import (
    Wav2Vec2VectorQuantizer as Wav2Vec2VectorQuantizer,
)
from fairseq2.models.wav2vec2.vector_quantizer import (
    Wav2Vec2VectorQuantizerOutput as Wav2Vec2VectorQuantizerOutput,
)
