# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Optional, Sequence, final

from torch import Tensor

from fairseq2 import _DOC_MODE
from fairseq2.data.text.text_tokenizer import TextTokenDecoder, TextTokenEncoder
from fairseq2.data.typing import PathLike, StringLike
from fairseq2.data.vocabulary_info import VocabularyInfo
from fairseq2.typing import Device, finaloverride

if TYPE_CHECKING or _DOC_MODE:

    @final
    class SentencePieceModel:
        def __init__(
            self,
            pathname: PathLike,
            control_symbols: Optional[Sequence[StringLike]] = None,
        ) -> None:
            ...

        def token_to_index(self, token: StringLike) -> int:
            ...

        def index_to_token(self, idx: int) -> str:
            ...

        @property
        def unk_idx(self) -> Optional[int]:
            ...

        @property
        def bos_idx(self) -> Optional[int]:
            ...

        @property
        def eos_idx(self) -> Optional[int]:
            ...

        @property
        def pad_idx(self) -> Optional[int]:
            ...

        @property
        def vocabulary_size(self) -> int:
            ...

    @final
    class SentencePieceEncoder(TextTokenEncoder):
        def __init__(
            self,
            model: SentencePieceModel,
            prefix_tokens: Optional[Sequence[StringLike]] = None,
            suffix_tokens: Optional[Sequence[StringLike]] = None,
            reverse: bool = False,
            enable_sampling: bool = False,
            nbest_size: int = -1,
            alpha: float = 0.1,
            device: Optional[Device] = None,
            pin_memory: bool = False,
        ) -> None:
            ...

        @finaloverride
        def __call__(self, sentence: StringLike) -> Tensor:
            ...

        @property
        @finaloverride
        def prefix_indices(self) -> Optional[Tensor]:
            ...

        @property
        @finaloverride
        def suffix_indices(self) -> Optional[Tensor]:
            ...

    @final
    class SentencePieceDecoder(TextTokenDecoder):
        def __init__(self, model: SentencePieceModel, reverse: bool = False) -> None:
            ...

        @finaloverride
        def __call__(self, token_indices: Tensor) -> List[StringLike]:
            ...

else:
    from fairseq2n.bindings.data.text.sentencepiece import (
        SentencePieceDecoder as SentencePieceDecoder,
    )
    from fairseq2n.bindings.data.text.sentencepiece import (
        SentencePieceEncoder as SentencePieceEncoder,
    )
    from fairseq2n.bindings.data.text.sentencepiece import (
        SentencePieceModel as SentencePieceModel,
    )

    # Ensure that extension types are virtual subclasses of their corresponding
    # abstract base types.
    TextTokenEncoder.register(SentencePieceEncoder)
    TextTokenDecoder.register(SentencePieceDecoder)

    def _set_module_name() -> None:
        for t in [SentencePieceDecoder, SentencePieceEncoder, SentencePieceModel]:
            t.__module__ = __name__

    _set_module_name()


def vocabulary_from_sentencepiece(model: SentencePieceModel) -> VocabularyInfo:
    """Return the vocabulary information of ``model``."""
    return VocabularyInfo(
        model.vocabulary_size,
        model.unk_idx,
        model.bos_idx,
        model.eos_idx,
        model.pad_idx,
    )
