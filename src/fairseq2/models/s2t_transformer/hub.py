# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.s2t_transformer.config import S2TTransformerConfig
from fairseq2.models.transformer import TransformerModel

s2t_transformer_hub = ModelHubAccessor(TransformerModel, S2TTransformerConfig)
