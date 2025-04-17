# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.transformer_lm._model import (
    TransformerLanguageModel as TransformerLanguageModel,
)
from fairseq2.models.transformer_lm._sharder import (
    shard_transformer_language_model as shard_transformer_language_model,
)
