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
from fairseq2.data.text.sentencepiece import (
    SentencePieceDecoder as SentencePieceDecoder,
)
from fairseq2.data.text.sentencepiece import (
    SentencePieceEncoder as SentencePieceEncoder,
)
from fairseq2.data.text.sentencepiece import SentencePieceModel as SentencePieceModel


class LineEnding(Enum):
    INFER = 0
    LF = 1
    CRLF = 2


def read_text(
    pathname: StringLike,
    encoding: StringLike = "",
    line_ending: LineEnding = LineEnding.INFER,
    ltrim: bool = False,
    rtrim: bool = False,
    skip_empty: bool = False,
    memory_map: bool = False,
    block_size: Optional[int] = None,
) -> DataPipelineBuilder:
    pass


if not TYPE_CHECKING and not DOC_MODE:
    from fairseq2.C.data.text import LineEnding, read_text  # noqa: F811

    def _set_module() -> None:
        for t in [LineEnding, read_text]:
            t.__module__ = __name__

    _set_module()
