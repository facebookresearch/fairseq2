# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.w2vbert.factory import W2VBERT_FAMILY as W2VBERT_FAMILY
from fairseq2.models.w2vbert.factory import W2VBertBuilder as W2VBertBuilder
from fairseq2.models.w2vbert.factory import W2VBertConfig as W2VBertConfig
from fairseq2.models.w2vbert.factory import create_w2vbert_model as create_w2vbert_model
from fairseq2.models.w2vbert.factory import w2vbert_arch as w2vbert_arch
from fairseq2.models.w2vbert.factory import w2vbert_archs as w2vbert_archs
from fairseq2.models.w2vbert.loader import load_w2vbert_config as load_w2vbert_config
from fairseq2.models.w2vbert.loader import load_w2vbert_model as load_w2vbert_model
from fairseq2.models.w2vbert.model import W2VBertLoss as W2VBertLoss
from fairseq2.models.w2vbert.model import W2VBertModel as W2VBertModel
from fairseq2.models.w2vbert.model import W2VBertOutput as W2VBertOutput

# isort: split

from fairseq2.dependency import DependencyContainer
from fairseq2.models.w2vbert.archs import register_archs


def register_w2vbert(container: DependencyContainer) -> None:
    register_archs()
