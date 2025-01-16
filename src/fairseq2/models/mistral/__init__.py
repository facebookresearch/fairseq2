# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.mistral._config import MISTRAL_MODEL_FAMILY as MISTRAL_MODEL_FAMILY
from fairseq2.models.mistral._config import MistralConfig as MistralConfig
from fairseq2.models.mistral._config import (
    register_mistral_configs as register_mistral_configs,
)
from fairseq2.models.mistral._factory import MistralFactory as MistralFactory
from fairseq2.models.mistral._handler import MistralModelHandler as MistralModelHandler
from fairseq2.models.mistral._handler import (
    convert_mistral_checkpoint as convert_mistral_checkpoint,
)

# isort: split

from fairseq2.models import ModelHubAccessor
from fairseq2.models.transformer_decoder import TransformerDecoderModel

get_mistral_model_hub = ModelHubAccessor(TransformerDecoderModel, MistralConfig)
