# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Optional, Sequence, Union, final

import torch
from torch import Tensor

from fairseq2 import DOC_MODE
from fairseq2.data.string import String, StringLike


@final
class SentencePieceModel:
    def __init__(
        self,
        pathname: StringLike,
        control_tokens: Optional[Sequence[StringLike]] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        reverse: bool = False,
    ) -> None:
        pass

    def token_to_index(self, token: StringLike) -> int:
        pass

    def index_to_token(self, idx: int) -> String:
        pass

    @property
    def unk_idx(self) -> int:
        pass

    @property
    def bos_idx(self) -> int:
        pass

    @property
    def eos_idx(self) -> int:
        pass

    @property
    def pad_idx(self) -> int:
        pass

    @property
    def vocabulary_size(self) -> int:
        pass


@final
class SentencePieceEncoder:
    def __init__(
        self,
        model: SentencePieceModel,
        enable_sampling: bool = False,
        nbest_size: int = -1,
        alpha: float = 0.1,
        batch_size: Optional[int] = None,
        pad_to_length: Optional[int] = None,
        pad_to_multiple: int = 1,
        lef_pad: bool = False,
        dtype=torch.int32,
        device=None,
        pin_memory: bool = False,
        disable_parallelism: bool = False,
    ) -> None:
        pass

    def __call__(self, sentences: Union[StringLike, Sequence[StringLike]]) -> Tensor:
        pass


@final
class SentencePieceDecoder:
    def __init__(self, model: SentencePieceModel) -> None:
        pass

    def __call__(self, token_indices: Tensor) -> List[String]:
        pass


if not TYPE_CHECKING and not DOC_MODE:
    from fairseq2._C.data.text.sentencepiece import (  # noqa: F811
        SentencePieceDecoder,
        SentencePieceEncoder,
        SentencePieceModel,
    )

    def _set_module() -> None:
        for t in [SentencePieceDecoder, SentencePieceEncoder, SentencePieceModel]:
            t.__module__ = __name__

    _set_module()
