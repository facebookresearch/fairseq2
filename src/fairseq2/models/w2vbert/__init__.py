# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.w2vbert.config import W2VBERT_MODEL_FAMILY as W2VBERT_MODEL_FAMILY
from fairseq2.models.w2vbert.config import W2VBertConfig as W2VBertConfig
from fairseq2.models.w2vbert.config import (
    register_w2vbert_configs as register_w2vbert_configs,
)
from fairseq2.models.w2vbert.factory import W2VBertFactory as W2VBertFactory
from fairseq2.models.w2vbert.handler import W2VBertModelHandler as W2VBertModelHandler
from fairseq2.models.w2vbert.handler import (
    convert_w2vbert_checkpoint as convert_w2vbert_checkpoint,
)
from fairseq2.models.w2vbert.model import W2VBertLoss as W2VBertLoss
from fairseq2.models.w2vbert.model import W2VBertModel as W2VBertModel
from fairseq2.models.w2vbert.model import W2VBertOutput as W2VBertOutput

# isort: split

from fairseq2.models.hub import ModelHubAccessor

get_w2vbert_model_hub = ModelHubAccessor(W2VBertModel, W2VBertConfig)
