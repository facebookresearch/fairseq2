# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.s2t_conformer.config import (
    S2T_CONFORMER_FAMILY,
    S2TConformerConfig,
)
from fairseq2.models.transformer import TransformerModel

get_s2t_conformer_model_hub = ModelHubAccessor(
    S2T_CONFORMER_FAMILY, TransformerModel, S2TConformerConfig
)
