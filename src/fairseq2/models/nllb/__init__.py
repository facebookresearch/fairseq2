# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.nllb.loader import load_nllb_tokenizer as load_nllb_tokenizer
from fairseq2.models.nllb.tokenizer import NllbTokenizer as NllbTokenizer

# isort: split

from fairseq2.dependency import DependencyContainer
from fairseq2.models.nllb.archs import register_archs


def register_nllb(container: DependencyContainer) -> None:
    register_archs()
