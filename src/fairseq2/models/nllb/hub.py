# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.data.tokenizers import TokenizerHubAccessor
from fairseq2.models.nllb.config import NLLB_FAMILY
from fairseq2.models.nllb.tokenizer import NllbTokenizerConfig

get_nllb_tokenizer_hub = TokenizerHubAccessor(NLLB_FAMILY, NllbTokenizerConfig)
