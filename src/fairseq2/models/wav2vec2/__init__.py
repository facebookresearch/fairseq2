# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2._config import (
    WAV2VEC2_MODEL_FAMILY as WAV2VEC2_MODEL_FAMILY,
)
from fairseq2.models.wav2vec2._config import Wav2Vec2Config as Wav2Vec2Config
from fairseq2.models.wav2vec2._config import (
    Wav2Vec2EncoderConfig as Wav2Vec2EncoderConfig,
)
from fairseq2.models.wav2vec2._config import (
    register_wav2vec2_configs as register_wav2vec2_configs,
)
from fairseq2.models.wav2vec2._factory import (
    Wav2Vec2EncoderFactory as Wav2Vec2EncoderFactory,
)
from fairseq2.models.wav2vec2._factory import Wav2Vec2Factory as Wav2Vec2Factory
from fairseq2.models.wav2vec2._feature_extractor import (
    Wav2Vec2FbankFeatureExtractor as Wav2Vec2FbankFeatureExtractor,
)
from fairseq2.models.wav2vec2._feature_extractor import (
    Wav2Vec2FeatureExtractor as Wav2Vec2FeatureExtractor,
)
from fairseq2.models.wav2vec2._frontend import Wav2Vec2Frontend as Wav2Vec2Frontend
from fairseq2.models.wav2vec2._handler import (
    Wav2Vec2ModelHandler as Wav2Vec2ModelHandler,
)
from fairseq2.models.wav2vec2._handler import (
    convert_wav2vec2_checkpoint as convert_wav2vec2_checkpoint,
)
from fairseq2.models.wav2vec2._masker import (
    StandardWav2Vec2Masker as StandardWav2Vec2Masker,
)
from fairseq2.models.wav2vec2._masker import Wav2Vec2Masker as Wav2Vec2Masker
from fairseq2.models.wav2vec2._masker import (
    extract_masked_elements as extract_masked_elements,
)
from fairseq2.models.wav2vec2._model import Wav2Vec2Features as Wav2Vec2Features
from fairseq2.models.wav2vec2._model import Wav2Vec2Loss as Wav2Vec2Loss
from fairseq2.models.wav2vec2._model import Wav2Vec2Model as Wav2Vec2Model
from fairseq2.models.wav2vec2._model import Wav2Vec2Output as Wav2Vec2Output
from fairseq2.models.wav2vec2._position_encoder import (
    Wav2Vec2PositionEncoder as Wav2Vec2PositionEncoder,
)
from fairseq2.models.wav2vec2._position_encoder import (
    Wav2Vec2StackedPositionEncoder as Wav2Vec2StackedPositionEncoder,
)
from fairseq2.models.wav2vec2._vector_quantizer import (
    GumbelVectorQuantizer as GumbelVectorQuantizer,
)
from fairseq2.models.wav2vec2._vector_quantizer import (
    GumbelVectorQuantizerOutput as GumbelVectorQuantizerOutput,
)
from fairseq2.models.wav2vec2._vector_quantizer import (
    VectorQuantizer as VectorQuantizer,
)
from fairseq2.models.wav2vec2._vector_quantizer import (
    VectorQuantizerOutput as VectorQuantizerOutput,
)

# isort: split

from fairseq2.models import ModelHubAccessor

get_wav2vec2_model_hub = ModelHubAccessor(Wav2Vec2Model, Wav2Vec2Config)
