# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["SentencePieceDecoder", "SentencePieceEncoder", "SentencePieceModel"]

from fairseq2._C.data.text.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
)


def _set_module() -> None:
    for t in [SentencePieceDecoder, SentencePieceEncoder, SentencePieceModel]:
        t.__module__ = __name__


_set_module()
