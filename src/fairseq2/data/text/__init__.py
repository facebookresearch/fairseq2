# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "LineEnding",
    "SentencePieceDecoder",
    "SentencePieceEncoder",
    "SentencePieceModel",
    "read_text",
]

from fairseq2._C.data.text import LineEnding, read_text
from fairseq2.data.text.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
)


def _set_module() -> None:
    for t in [LineEnding, read_text]:
        t.__module__ = __name__


_set_module()
