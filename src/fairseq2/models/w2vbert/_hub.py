# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor

# isort: split

from fairseq2.models.w2vbert._config import W2VBertConfig
from fairseq2.models.w2vbert._model import W2VBertModel

get_w2vbert_model_hub = ModelHubAccessor(W2VBertModel, W2VBertConfig)
