# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.transformer.config import TRANSFORMER_FAMILY, TransformerConfig
from fairseq2.models.transformer.model import TransformerModel

get_transformer_model_hub = ModelHubAccessor(
    TRANSFORMER_FAMILY, TransformerModel, TransformerConfig
)
