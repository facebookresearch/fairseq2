# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.w2vbert.config import W2VBertConfig
from fairseq2.models.w2vbert.model import W2VBertModel

w2vbert_hub = ModelHubAccessor(W2VBertModel, W2VBertConfig)
