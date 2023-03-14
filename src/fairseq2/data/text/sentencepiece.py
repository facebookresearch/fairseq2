# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Optional, Sequence, Union, final

import torch
from torch import Tensor

from fairseq2 import DOC_MODE
from fairseq2.data.string import StringLike
from fairseq2.data.text.tokenizer import TokenDecoder, TokenEncoder
from fairseq2.data.typing import PathLike


@final
class SentencePieceModel:
    def __init__(
        self, pathname: PathLike, control_tokens: Optional[Sequence[StringLike]] = None
    ) -> None:
        pass

    def token_to_index(self, token: StringLike) -> int:
        return 0

    def index_to_token(self, idx: int) -> str:
        return ""

    @property
    def unk_idx(self) -> int:
        return 0

    @property
    def bos_idx(self) -> int:
        return 0

    @property
    def eos_idx(self) -> int:
        return 0

    @property
    def pad_idx(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return 0


@final
class SentencePieceEncoder(TokenEncoder):
    def __init__(
        self,
        model: SentencePieceModel,
        prefix_tokens: Optional[Sequence[StringLike]] = None,
        suffix_tokens: Optional[Sequence[StringLike]] = None,
        reverse: bool = False,
        enable_sampling: bool = False,
        nbest_size: int = -1,
        alpha: float = 0.1,
        batch_size: Optional[int] = None,
        pad_to_length: Optional[int] = None,
        pad_to_multiple: int = 1,
        left_pad: bool = False,
        dtype: torch.dtype = torch.int32,
        device: Optional[torch.device] = None,
        pin_memory: bool = False,
        disable_parallelism: bool = False,
    ) -> None:
        pass

    def __call__(self, sentences: Union[StringLike, Sequence[StringLike]]) -> Tensor:
        raise NotImplementedError()


@final
class SentencePieceDecoder(TokenDecoder):
    def __init__(self, model: SentencePieceModel, reverse: bool = False) -> None:
        pass

    def __call__(self, token_indices: Tensor) -> Union[StringLike, List[StringLike]]:
        raise NotImplementedError()


if not TYPE_CHECKING and not DOC_MODE:
    from fairseq2.C.data.text.sentencepiece import (  # noqa: F811
        SentencePieceDecoder,
        SentencePieceEncoder,
        SentencePieceModel,
    )

    # Ensure that extension types are virtual subclasses of their corresponding
    # abstract base types.
    TokenEncoder.register(SentencePieceEncoder)
    TokenDecoder.register(SentencePieceDecoder)

    def _set_module() -> None:
        for t in [SentencePieceDecoder, SentencePieceEncoder, SentencePieceModel]:
            t.__module__ = __name__

    _set_module()
