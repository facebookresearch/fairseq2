# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.s2t_transformer._config import (
    S2T_TRANSFORMER_MODEL_FAMILY as S2T_TRANSFORMER_MODEL_FAMILY,
)
from fairseq2.models.s2t_transformer._config import (
    S2TTransformerConfig as S2TTransformerConfig,
)
from fairseq2.models.s2t_transformer._config import (
    register_s2t_transformer_configs as register_s2t_transformer_configs,
)
from fairseq2.models.s2t_transformer._factory import (
    S2TTransformerFactory as S2TTransformerFactory,
)
from fairseq2.models.s2t_transformer._feature_extractor import (
    Conv1dFbankSubsampler as Conv1dFbankSubsampler,
)
from fairseq2.models.s2t_transformer._frontend import (
    S2TTransformerFrontend as S2TTransformerFrontend,
)
from fairseq2.models.s2t_transformer._handler import (
    S2TTransformerModelHandler as S2TTransformerModelHandler,
)
from fairseq2.models.s2t_transformer._handler import (
    convert_s2t_transformer_checkpoint as convert_s2t_transformer_checkpoint,
)

# isort: split

from fairseq2.models import ModelHubAccessor
from fairseq2.models.transformer import TransformerModel

get_s2t_transformer_model_hub = ModelHubAccessor(TransformerModel, S2TTransformerConfig)
