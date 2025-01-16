# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.transformer_decoder._model import (
    TransformerDecoderModel as TransformerDecoderModel,
)
from fairseq2.models.transformer_decoder._sharder import (
    shard_transformer_decoder_model as shard_transformer_decoder_model,
)
