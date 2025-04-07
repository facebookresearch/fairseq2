# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.qwen25._checkpoint import (
    convert_qwen_checkpoint as convert_qwen_checkpoint,
)
from fairseq2.models.qwen25._config import QWEN25_MODEL_FAMILY as QWEN25_MODEL_FAMILY
from fairseq2.models.qwen25._config import Qwen25Config as Qwen25Config
from fairseq2.models.qwen25._config import (
    register_qwen_configs as register_qwen_configs,
)
from fairseq2.models.qwen25._factory import Qwen25Factory as Qwen25Factory
from fairseq2.models.qwen25._factory import create_qwen25_model as create_qwen25_model
from fairseq2.models.qwen25._shard import shard_qwen_model as shard_qwen_model

# isort: split

from fairseq2.models import ModelHubAccessor
from fairseq2.models.transformer_decoder import TransformerDecoderModel

get_llama_model_hub = ModelHubAccessor(TransformerDecoderModel, Qwen25Config)
