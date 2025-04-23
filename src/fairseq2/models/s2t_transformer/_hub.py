# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.transformer import TransformerModel

# isort: split

from fairseq2.models.s2t_transformer._config import S2TTransformerConfig

get_s2t_transformer_model_hub = ModelHubAccessor(TransformerModel, S2TTransformerConfig)
