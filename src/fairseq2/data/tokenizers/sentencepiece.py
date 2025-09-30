# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, final

from fairseq2n import DOC_MODE
from torch import Tensor
from typing_extensions import override

from fairseq2.data.tokenizers.family import TokenizerModelError
from fairseq2.data.tokenizers.tokenizer import TokenDecoder, TokenEncoder, Tokenizer
from fairseq2.data.tokenizers.vocab_info import VocabularyInfo
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.runtime.dependency import get_dependency_resolver

if TYPE_CHECKING or DOC_MODE:

    @final
    class SentencePieceModel:
        def __init__(
            self, path: Path, control_symbols: Sequence[str] | None = None
        ) -> None: ...

        def token_to_index(self, token: str) -> int: ...

        def index_to_token(self, idx: int) -> str: ...

        @property
        def unk_idx(self) -> int | None: ...

        @property
        def bos_idx(self) -> int | None: ...

        @property
        def eos_idx(self) -> int | None: ...

        @property
        def pad_idx(self) -> int | None: ...

        @property
        def vocabulary_size(self) -> int: ...

    @final
    class SentencePieceEncoder(TokenEncoder):
        def __init__(
            self,
            model: SentencePieceModel,
            prefix_tokens: Sequence[str] | None = None,
            suffix_tokens: Sequence[str] | None = None,
            reverse: bool = False,
            enable_sampling: bool = False,
            nbest_size: int = -1,
            alpha: float = 0.1,
            device: Device | None = None,
            pin_memory: bool = False,
        ) -> None: ...

        @override
        def __call__(self, text: str) -> Tensor: ...

        @override
        def encode_as_tokens(self, text: str) -> list[str]: ...

        @property
        @override
        def prefix_indices(self) -> Tensor | None: ...

        @property
        @override
        def suffix_indices(self) -> Tensor | None: ...

    @final
    class SentencePieceDecoder(TokenDecoder):
        def __init__(
            self, model: SentencePieceModel, reverse: bool = False
        ) -> None: ...

        @override
        def __call__(self, token_indices: Tensor) -> str: ...

        @override
        def decode_from_tokens(self, tokens: Sequence[str]) -> str: ...

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
    TokenEncoder.register(SentencePieceEncoder)
    TokenDecoder.register(SentencePieceDecoder)

    def _set_module_name() -> None:
        for t in [SentencePieceDecoder, SentencePieceEncoder, SentencePieceModel]:
            t.__module__ = __name__

    _set_module_name()


@final
class BasicSentencePieceTokenizer(Tokenizer):
    """Represents a SentencePiece tokenizer that encodes text with BOS and EOS."""

    def __init__(self, model: SentencePieceModel) -> None:
        self._model = model

        self._vocab_info = get_sentencepiece_vocabulary_info(model)

    @override
    def create_encoder(
        self,
        *,
        task: str | None = None,
        lang: str | None = None,
        mode: str | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> TokenEncoder:
        """Constructs a token encoder.

        :param task:
            Must be ``None``.
        :param lang:
            Must be ``None``.
        :param mode:
            Must be 'default', 'prompt', or 'prompt_response'. If ``None``,
            defaults to 'default'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")

        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")

        match mode:
            case None | "default":
                prefix_tokens = ["<s>"]
                suffix_tokens = ["</s>"]
            case "prompt":
                prefix_tokens = ["<s>"]
                # In prompt mode, we expect the generator to finish the sequence.
                suffix_tokens = []
            case "prompt_response":
                prefix_tokens = []
                suffix_tokens = ["</s>"]
            case _:
                raise NotSupportedError(
                    f"`mode` must be a supported mode, but is {mode} instead. Supported modes are default, prompt, prompt_response."
                )

        return SentencePieceEncoder(
            self._model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return SentencePieceEncoder(self._model, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return SentencePieceDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


@final
class RawSentencePieceTokenizer(Tokenizer):
    """Represents a SentencePiece tokenizer that encodes text with no control symbols."""

    def __init__(self, model: SentencePieceModel) -> None:
        self._model = model

        self._vocab_info = get_sentencepiece_vocabulary_info(model)

    @override
    def create_encoder(
        self,
        *,
        task: str | None = None,
        lang: str | None = None,
        mode: str | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> TokenEncoder:
        """Constructs a token encoder.

        :param task:
            Must be ``None``.
        :param lang:
            Must be ``None``.
        :param mode:
            Must be ``None``.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")

        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")

        if mode is not None:
            raise ValueError(f"`mode` must be `None`, but is '{mode}' instead.")

        return self.create_raw_encoder(device=device, pin_memory=pin_memory)

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return SentencePieceEncoder(self._model, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return SentencePieceDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


class SentencePieceModelLoader(ABC):
    @abstractmethod
    def load(
        self, path: Path, *, control_symbols: Sequence[str] | None = None
    ) -> SentencePieceModel: ...


@final
class StandardSentencePieceModelLoader(SentencePieceModelLoader):
    @override
    def load(
        self, path: Path, *, control_symbols: Sequence[str] | None = None
    ) -> SentencePieceModel:
        try:
            return SentencePieceModel(path, control_symbols)
        except RuntimeError as ex:
            msg = f"{path} is not a valid SentencePiece model."

            raise TokenizerModelError(path, msg) from ex


def load_sentencepiece_model(
    path: Path, *, control_symbols: Sequence[str] | None = None
) -> SentencePieceModel:
    resolver = get_dependency_resolver()

    model_loader = resolver.resolve(SentencePieceModelLoader)

    return model_loader.load(path, control_symbols=control_symbols)


def get_sentencepiece_vocabulary_info(model: SentencePieceModel) -> VocabularyInfo:
    """Return the vocabulary information of ``model``."""
    return VocabularyInfo(
        model.vocabulary_size,
        model.unk_idx,
        model.bos_idx,
        model.eos_idx,
        model.pad_idx,
    )
