# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Final, final

from typing_extensions import override

from fairseq2.assets import AssetCard, AssetCardError, AssetCardFieldNotFoundError
from fairseq2.data import VocabularyInfo
from fairseq2.data.text.tokenizers import (
    TextTokenDecoder,
    TextTokenEncoder,
    TextTokenizer,
    TextTokenizerLoadError,
    text_tokenizer_asset_card_error,
)
from fairseq2.data.text.tokenizers.hg import (
    HuggingFaceTokenDecoder,
    HuggingFaceTokenEncoder,
    HuggingFaceTokenModel,
    load_hg_token_model,
)
from fairseq2.device import Device


@final
class QwenTokenizer(TextTokenizer):
    """Represents a Qwen tokenizer"""

    _model: HuggingFaceTokenModel
    _eos_token: str

    def __init__(self, model: HuggingFaceTokenModel, eos_token: str) -> None:
        self._model = model

        self._eos_token = eos_token

    @override
    def create_encoder(
        self,
        *,
        task: str | None = None,
        lang: str | None = None,
        mode: str | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> TextTokenEncoder:
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")

        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")

        match mode:
            case None | "default":
                suffix_tokens = [self._eos_token]
            case "prompt":
                # In prompt mode, we expect the generator to finish the sequence.
                suffix_tokens = []
            case "prompt_response":
                suffix_tokens = [self._eos_token]
            case "as_is":
                suffix_tokens = []
            case _:
                raise ValueError(
                    f"`mode` must be one of the following values, but is '{mode}' instead: default, prompt, prompt_response, as_is"
                )

        return HuggingFaceTokenEncoder(
            self._model,
            prefix_tokens=[],
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TextTokenEncoder:
        return HuggingFaceTokenEncoder(
            self._model, device=device, pin_memory=pin_memory
        )

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TextTokenDecoder:
        return HuggingFaceTokenDecoder(
            self._model, skip_special_tokens=skip_special_tokens
        )

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._model.vocab_info


QWEN_TOKENIZER_FAMILY: Final = "qwen"


def load_qwen_tokenizer(path: Path, card: AssetCard) -> TextTokenizer:
    try:
        use_im_end = card.field("use_im_end").as_(bool)
    except AssetCardFieldNotFoundError:
        use_im_end = False
    except AssetCardError as ex:
        raise text_tokenizer_asset_card_error(card.name) from ex

    eos_token = "<|im_end|>" if use_im_end else "<|endoftext|>"

    try:
        model = load_hg_token_model(
            path,
            unk_token=None,
            bos_token=None,
            eos_token=eos_token,
            pad_token="<|endoftext|>",
            boh_token=None,
            eoh_token=None,
        )
    except (OSError, RuntimeError) as ex:
        raise TextTokenizerLoadError(
            card.name, f"The '{card.name}' text tokenizer model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    return QwenTokenizer(model, eos_token)
