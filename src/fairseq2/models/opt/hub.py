# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor
from fairseq2.models.transformer_lm import TransformerLM

# isort: split

from fairseq2.models.opt.config import OPTConfig, OPT_MODEL_FAMILY

get_opt_model_hub = ModelHubAccessor(OPT_MODEL_FAMILY, TransformerLM, OPTConfig)
