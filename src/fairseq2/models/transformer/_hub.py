# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor

# isort: split

from fairseq2.models.transformer._config import TransformerConfig
from fairseq2.models.transformer._model import TransformerModel

get_transformer_model_hub = ModelHubAccessor(TransformerModel, TransformerConfig)
