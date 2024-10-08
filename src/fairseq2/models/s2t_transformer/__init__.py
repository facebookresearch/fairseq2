# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.s2t_transformer.factory import (
    S2T_TRANSFORMER_FAMILY as S2T_TRANSFORMER_FAMILY,
)
from fairseq2.models.s2t_transformer.factory import (
    S2TTransformerBuilder as S2TTransformerBuilder,
)
from fairseq2.models.s2t_transformer.factory import (
    S2TTransformerConfig as S2TTransformerConfig,
)
from fairseq2.models.s2t_transformer.factory import (
    create_s2t_transformer_model as create_s2t_transformer_model,
)
from fairseq2.models.s2t_transformer.factory import (
    s2t_transformer_arch as s2t_transformer_arch,
)
from fairseq2.models.s2t_transformer.factory import (
    s2t_transformer_archs as s2t_transformer_archs,
)
from fairseq2.models.s2t_transformer.feature_extractor import (
    Conv1dFbankSubsampler as Conv1dFbankSubsampler,
)
from fairseq2.models.s2t_transformer.frontend import (
    S2TTransformerFrontend as S2TTransformerFrontend,
)
from fairseq2.models.s2t_transformer.loader import (
    load_s2t_transformer_config as load_s2t_transformer_config,
)
from fairseq2.models.s2t_transformer.loader import (
    load_s2t_transformer_model as load_s2t_transformer_model,
)
from fairseq2.models.s2t_transformer.loader import (
    load_s2t_transformer_tokenizer as load_s2t_transformer_tokenizer,
)
from fairseq2.models.s2t_transformer.tokenizer import (
    S2TTransformerTokenizer as S2TTransformerTokenizer,
)

# isort: split

from fairseq2.dependency import DependencyContainer
from fairseq2.models.s2t_transformer.archs import register_archs


def register_s2t_transformer(container: DependencyContainer) -> None:
    register_archs()
