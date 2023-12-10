# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Optional, Sequence, final

from fairseq2n import DOC_MODE
from torch import Tensor

from fairseq2.data.text.text_tokenizer import (
    TextTokenDecoder,
    TextTokenEncoder,
    TextTokenizer,
)
from fairseq2.data.typing import PathLike, StringLike
from fairseq2.data.vocabulary_info import VocabularyInfo
from fairseq2.typing import Device, finaloverride

if TYPE_CHECKING or DOC_MODE:

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
        def __call__(self, text: StringLike) -> Tensor:
            ...

        @finaloverride
        def encode_as_tokens(self, text: StringLike) -> List[StringLike]:
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
        def __call__(self, token_indices: Tensor) -> StringLike:
            ...

        @finaloverride
        def decode_from_tokens(self, tokens: Sequence[StringLike]) -> StringLike:
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


class SentencePieceTokenizerBase(TextTokenizer):
    """Represents an abstract base class for SentencePiece tokenizers."""

    model: SentencePieceModel

    def __init__(
        self, pathname: PathLike, control_symbols: Optional[Sequence[StringLike]] = None
    ) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        :param control_symbols:
            The list of control symbols to add to the SentencePiece model.
        """
        self.model = SentencePieceModel(pathname, control_symbols)

        vocab_info = vocab_info_from_sentencepiece(self.model)

        super().__init__(vocab_info)

    @finaloverride
    def create_raw_encoder(
        self, *, device: Optional[Device] = None, pin_memory: bool = False
    ) -> SentencePieceEncoder:
        return SentencePieceEncoder(self.model, device=device, pin_memory=pin_memory)

    @finaloverride
    def create_decoder(self) -> SentencePieceDecoder:
        return SentencePieceDecoder(self.model)


class BasicSentencePieceTokenizer(SentencePieceTokenizerBase):
    """Represents a SentencePiece tokenizer that encodes text with BOS and EOS."""

    def __init__(self, pathname: PathLike) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        """
        super().__init__(pathname)

    @finaloverride
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> SentencePieceEncoder:
        """Create a token encoder.

        :param task:
            Not used.
        :param lang:
            Not used.
        :param mode:
            Must be 'default' or 'prompt'. If ``None``, defaults to 'default'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")

        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")

        if mode is None or mode == "default":
            prefix_tokens = ["<s>"]
            suffix_tokens = ["</s>"]
        elif mode == "prompt":
            prefix_tokens = ["<s>"]
            # In prompt mode, we expect the generator to finish the sequence.
            suffix_tokens = None
        else:
            raise ValueError(
                f"`mode` must be 'default' or 'prompt', but is '{mode}' instead."
            )

        return SentencePieceEncoder(
            self.model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )


def vocab_info_from_sentencepiece(model: SentencePieceModel) -> VocabularyInfo:
    """Return the vocabulary information of ``model``."""
    return VocabularyInfo(
        model.vocabulary_size,
        model.unk_idx,
        model.bos_idx,
        model.eos_idx,
        model.pad_idx,
    )
