# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.nllb.factory import NLLB_FAMILY as NLLB_FAMILY
from fairseq2.models.nllb.factory import NllbBuilder as NllbBuilder
from fairseq2.models.nllb.factory import NllbConfig as NllbConfig
from fairseq2.models.nllb.factory import create_nllb_model as create_nllb_model
from fairseq2.models.nllb.factory import nllb_arch as nllb_arch
from fairseq2.models.nllb.factory import nllb_archs as nllb_archs
from fairseq2.models.nllb.setup import load_nllb_config as load_nllb_config
from fairseq2.models.nllb.setup import load_nllb_model as load_nllb_model
from fairseq2.models.nllb.setup import load_nllb_tokenizer as load_nllb_tokenizer
from fairseq2.models.nllb.tokenizer import NllbTokenizer as NllbTokenizer

# isort: split

from fairseq2.data.text import load_text_tokenizer
from fairseq2.models.loader import load_model


def _register_nllb() -> None:
    load_model.register(NLLB_FAMILY, load_nllb_model)

    load_text_tokenizer.register(NLLB_FAMILY, load_nllb_tokenizer)
