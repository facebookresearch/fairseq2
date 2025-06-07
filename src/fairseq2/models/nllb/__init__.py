# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.nllb.config import NLLB_FAMILY as NLLB_FAMILY
from fairseq2.models.nllb.config import _register_nllb_configs as _register_nllb_configs
from fairseq2.models.nllb.hub import get_nllb_tokenizer_hub as get_nllb_tokenizer_hub
from fairseq2.models.nllb.tokenizer import NllbTokenizer as NllbTokenizer
from fairseq2.models.nllb.tokenizer import NllbTokenizerConfig as NllbTokenizerConfig
from fairseq2.models.nllb.tokenizer import _load_nllb_tokenizer as _load_nllb_tokenizer
