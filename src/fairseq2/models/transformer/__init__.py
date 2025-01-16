# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.transformer._config import (
    TRANSFORMER_MODEL_FAMILY as TRANSFORMER_MODEL_FAMILY,
)
from fairseq2.models.transformer._config import TransformerConfig as TransformerConfig
from fairseq2.models.transformer._config import (
    register_transformer_configs as register_transformer_configs,
)
from fairseq2.models.transformer._factory import (
    TransformerFactory as TransformerFactory,
)
from fairseq2.models.transformer._frontend import (
    TransformerEmbeddingFrontend as TransformerEmbeddingFrontend,
)
from fairseq2.models.transformer._frontend import (
    TransformerFrontend as TransformerFrontend,
)
from fairseq2.models.transformer._handler import (
    TransformerModelHandler as TransformerModelHandler,
)
from fairseq2.models.transformer._handler import (
    convert_transformer_checkpoint as convert_transformer_checkpoint,
)
from fairseq2.models.transformer._model import TransformerModel as TransformerModel
from fairseq2.models.transformer._model import (
    init_final_projection as init_final_projection,
)

# isort: split

from fairseq2.models import ModelHubAccessor

get_transformer_model_hub = ModelHubAccessor(TransformerModel, TransformerConfig)
