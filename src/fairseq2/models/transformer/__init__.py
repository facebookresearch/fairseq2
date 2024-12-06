# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.transformer.decoder_model import (
    TransformerDecoderModel as TransformerDecoderModel,
)
from fairseq2.models.transformer.decoder_model import (
    shard_transformer_decoder_model as shard_transformer_decoder_model,
)
from fairseq2.models.transformer.factory import TRANSFORMER_FAMILY as TRANSFORMER_FAMILY
from fairseq2.models.transformer.factory import TransformerBuilder as TransformerBuilder
from fairseq2.models.transformer.factory import TransformerConfig as TransformerConfig
from fairseq2.models.transformer.factory import (
    create_transformer_model as create_transformer_model,
)
from fairseq2.models.transformer.factory import transformer_arch as transformer_arch
from fairseq2.models.transformer.factory import transformer_archs as transformer_archs
from fairseq2.models.transformer.frontend import (
    TransformerEmbeddingFrontend as TransformerEmbeddingFrontend,
)
from fairseq2.models.transformer.frontend import (
    TransformerFrontend as TransformerFrontend,
)
from fairseq2.models.transformer.loader import (
    load_transformer_config as load_transformer_config,
)
from fairseq2.models.transformer.loader import (
    load_transformer_model as load_transformer_model,
)
from fairseq2.models.transformer.model import TransformerModel as TransformerModel
from fairseq2.models.transformer.model import (
    init_final_projection as init_final_projection,
)

# isort: split

import fairseq2.models.transformer.archs  # Register architectures.
