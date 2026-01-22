# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.w2vbert.config import W2VBERT_FAMILY as W2VBERT_FAMILY
from fairseq2.models.w2vbert.config import W2VBertConfig as W2VBertConfig
from fairseq2.models.w2vbert.config import (
    register_w2vbert_configs as register_w2vbert_configs,
)
from fairseq2.models.w2vbert.factory import W2VBertFactory as W2VBertFactory
from fairseq2.models.w2vbert.factory import create_w2vbert_model as create_w2vbert_model
from fairseq2.models.w2vbert.hub import get_w2vbert_model_hub as get_w2vbert_model_hub
from fairseq2.models.w2vbert.interop import (
    convert_w2vbert_state_dict as convert_w2vbert_state_dict,
)
from fairseq2.models.w2vbert.model import W2VBertLoss as W2VBertLoss
from fairseq2.models.w2vbert.model import W2VBertModel as W2VBertModel
from fairseq2.models.w2vbert.model import W2VBertOutput as W2VBertOutput
