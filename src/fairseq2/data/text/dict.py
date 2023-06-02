# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# TODO Find a better name for this file


from typing import TYPE_CHECKING, List, Sequence, Union

from overrides import final
from torch import Tensor

from fairseq2 import DOC_MODE
from fairseq2.data.string import StringLike
from fairseq2.data.text.tokenizer import TokenDecoder, TokenEncoder


@final
class DictModel:
    def __init__(self, vocab: Sequence[StringLike]) -> None:
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
class DictEncoder(TokenEncoder):
    def __init__(self, processor: DictModel, dim: int) -> None:
        pass

    def __call__(self, sentence: Union[StringLike, Sequence[StringLike]]) -> Tensor:
        raise NotImplementedError()


@final
class DictDecoder(TokenDecoder):
    def __init__(self, processor: DictModel) -> None:
        pass

    def __call__(self, token_indices: Tensor) -> List[StringLike]:
        raise NotImplementedError()


if not TYPE_CHECKING and not DOC_MODE:
    from fairseq2.C.data.text.dict import (  # noqa: F811
        DictDecoder,
        DictEncoder,
        DictModel,
    )

    # Ensure that extension types are virtual subclasses of their corresponding
    # abstract base types.
    TokenEncoder.register(DictEncoder)
    TokenDecoder.register(DictDecoder)

    def _set_module() -> None:
        for t in [DictEncoder, DictDecoder]:
            t.__module__ = __name__

    _set_module()
