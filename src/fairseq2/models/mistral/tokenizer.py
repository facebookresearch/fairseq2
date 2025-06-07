# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

from fairseq2.data.tokenizers import Tokenizer, TokenizerLoadError
from fairseq2.data.tokenizers.sentencepiece import (
    BasicSentencePieceTokenizer,
    SentencePieceModel,
)
from fairseq2.error import InfraError
from fairseq2.runtime.dependency import DependencyResolver


def _load_mistral_tokenizer(
    resolver: DependencyResolver, path: Path, name: str, config: None = None
) -> Tokenizer:
    try:
        model = SentencePieceModel(path)
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while reading the '{path}' tokenizer model. See the nested exception for details."
        ) from ex
    except RuntimeError as ex:
        raise TokenizerLoadError(
            name, f"The '{path}' tokenizer model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    return BasicSentencePieceTokenizer(model)
