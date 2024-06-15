# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.nllb.loader import load_nllb_tokenizer as load_nllb_tokenizer
from fairseq2.models.nllb.tokenizer import NllbTokenizer as NllbTokenizer

# isort: split

from fairseq2.data.text import load_text_tokenizer
from fairseq2.models.nllb.archs import _register_nllb_archs


def _register_nllb() -> None:
    _register_nllb_archs()

    load_text_tokenizer.register("nllb", load_nllb_tokenizer)


# isort: split

# compat
from fairseq2.models.transformer import load_transformer_model

load_nllb_model = load_transformer_model
