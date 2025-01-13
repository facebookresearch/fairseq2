# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.config_loader import ModelConfigLoader as ModelConfigLoader
from fairseq2.models.config_loader import (
    StandardModelConfigLoader as StandardModelConfigLoader,
)
from fairseq2.models.config_loader import get_model_family as get_model_family
from fairseq2.models.config_loader import is_model_card as is_model_card
from fairseq2.models.factory import create_model as create_model
from fairseq2.models.factory import model_factories as model_factories
from fairseq2.models.loader import CheckpointConverter as CheckpointConverter
from fairseq2.models.loader import DelegatingModelLoader as DelegatingModelLoader
from fairseq2.models.loader import ModelFactory as ModelFactory
from fairseq2.models.loader import ModelLoader as ModelLoader
from fairseq2.models.loader import StandardModelLoader as StandardModelLoader
from fairseq2.models.loader import load_model as load_model
from fairseq2.models.model import Model as Model

# isort: split

import fairseq2.models.llama
import fairseq2.models.mistral
import fairseq2.models.nllb
import fairseq2.models.s2t_transformer
import fairseq2.models.transformer
import fairseq2.models.w2vbert
import fairseq2.models.wav2vec2
import fairseq2.models.wav2vec2.asr
