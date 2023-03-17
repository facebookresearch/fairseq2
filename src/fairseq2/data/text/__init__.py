# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import TYPE_CHECKING, Optional

from fairseq2 import DOC_MODE
from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.data.string import StringLike
from fairseq2.data.text.read import LineEnding as LineEnding
from fairseq2.data.text.read import read_text as read_text
from fairseq2.data.text.sentencepiece import (
    SentencePieceDecoder as SentencePieceDecoder,
)
from fairseq2.data.text.sentencepiece import (
    SentencePieceEncoder as SentencePieceEncoder,
)
from fairseq2.data.text.sentencepiece import SentencePieceModel as SentencePieceModel
from fairseq2.data.text.tokenizer import TokenDecoder as TokenDecoder
from fairseq2.data.text.tokenizer import TokenEncoder as TokenEncoder
from fairseq2.data.text.tokenizer import Tokenizer as Tokenizer
from fairseq2.data.text.tokenizer import VocabularyInfo as VocabularyInfo
from fairseq2.data.typing import PathLike
