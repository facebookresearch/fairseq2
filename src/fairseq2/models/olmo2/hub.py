# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.olmo2.config import OLMO2_FAMILY, OLMO2Config
from fairseq2.models.transformer_lm import TransformerLM

get_olmo2_model_hub = ModelHubAccessor(
    OLMO2_FAMILY, kls=TransformerLM, config_kls=OLMO2Config
)
