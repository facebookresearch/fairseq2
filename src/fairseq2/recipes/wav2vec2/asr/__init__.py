# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipes.wav2vec2.asr._common import (
    Wav2Vec2AsrCriterion as Wav2Vec2AsrCriterion,
)
from fairseq2.recipes.wav2vec2.asr._common import (
    Wav2Vec2AsrMetricBag as Wav2Vec2AsrMetricBag,
)
from fairseq2.recipes.wav2vec2.asr._common import Wav2Vec2AsrScorer as Wav2Vec2AsrScorer
from fairseq2.recipes.wav2vec2.asr._eval import (
    Wav2Vec2AsrEvalConfig as Wav2Vec2AsrEvalConfig,
)
from fairseq2.recipes.wav2vec2.asr._eval import (
    Wav2Vec2AsrEvalUnit as Wav2Vec2AsrEvalUnit,
)
from fairseq2.recipes.wav2vec2.asr._eval import (
    load_wav2vec2_asr_evaluator as load_wav2vec2_asr_evaluator,
)
from fairseq2.recipes.wav2vec2.asr._eval import (
    register_wav2vec2_asr_eval_configs as register_wav2vec2_asr_eval_configs,
)
from fairseq2.recipes.wav2vec2.asr._train import (
    Wav2Vec2AsrTrainConfig as Wav2Vec2AsrTrainConfig,
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
