# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Optional, Sequence, final

from overrides import final as finaloverride
from torch import Tensor

from fairseq2 import _DOC_MODE
from fairseq2.data.text.text_tokenizer import (
    TextTokenDecoder,
    TextTokenEncoder,
    TextTokenizer,
)
from fairseq2.data.typing import StringLike
from fairseq2.data.vocabulary_info import VocabularyInfo
from fairseq2.typing import Device

if TYPE_CHECKING or _DOC_MODE:

    @final
    class DictModel:
        def __init__(self, vocab: Sequence[StringLike]) -> None:
            ...

        def token_to_index(self, token: StringLike) -> int:
            ...

        def index_to_token(self, idx: int) -> str:
            ...

        @property
        def unk_idx(self) -> int:
            ...

        @property
        def bos_idx(self) -> int:
            ...

        @property
        def eos_idx(self) -> int:
            ...

        @property
        def pad_idx(self) -> int:
            ...

        @property
        def vocab_size(self) -> int:
            ...

    @final
    class DictEncoder(TextTokenEncoder):
        def __init__(self, processor: DictModel, dim: int) -> None:
            ...

        def __call__(self, sentence: StringLike) -> Tensor:
            ...

    @final
    class DictDecoder(TextTokenDecoder):
        def __init__(self, processor: DictModel) -> None:
            ...

        def __call__(self, token_indices: Tensor) -> List[StringLike]:
            ...

else:
    from fairseq2.C.data.text.dict_tokenizer import DictDecoder as DictDecoder
    from fairseq2.C.data.text.dict_tokenizer import DictEncoder as DictEncoder
    from fairseq2.C.data.text.dict_tokenizer import DictModel as DictModel

    # Ensure that extension types are virtual subclasses of their corresponding
    # abstract base types.
    TextTokenEncoder.register(DictEncoder)
    TextTokenDecoder.register(DictDecoder)

    def _set_module_name() -> None:
        for t in [DictEncoder, DictDecoder, DictModel]:
            t.__module__ = __name__

    _set_module_name()


@final
class DictTokenizer(TextTokenizer):
    """Represents a simple tokenizer that splits on space and replace word by
    token found in dict."""

    model: DictModel
    dim: int

    def __init__(self, dim: int, vocab: Sequence[StringLike]) -> None:
        self.dim = dim
        self.model = DictModel(vocab)

        vocabulary_info = VocabularyInfo(
            self.model.vocab_size,
            self.model.unk_idx,
            self.model.bos_idx,
            self.model.eos_idx,
            self.model.pad_idx,
        )

        super().__init__(vocabulary_info)

    @finaloverride
    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> "TextTokenEncoder":
        return DictEncoder(self.model, self.dim)

    @finaloverride
    def create_decoder(self) -> "TextTokenDecoder":
        return DictDecoder(self.model)
