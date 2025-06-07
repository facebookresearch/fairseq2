# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import final

from typing_extensions import override

from fairseq2.data.tokenizers import (
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    TokenizerLoadError,
    VocabularyInfo,
)
from fairseq2.data.tokenizers.hg import (
    HuggingFaceTokenDecoder,
    HuggingFaceTokenEncoder,
    HuggingFaceTokenModel,
    load_hg_token_model,
)
from fairseq2.device import Device
from fairseq2.error import InfraError
from fairseq2.runtime.dependency import DependencyResolver


@final
class QwenTokenizer(Tokenizer):
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
    ) -> TokenEncoder:
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
    ) -> TokenEncoder:
        return HuggingFaceTokenEncoder(
            self._model, device=device, pin_memory=pin_memory
        )

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return HuggingFaceTokenDecoder(
            self._model, skip_special_tokens=skip_special_tokens
        )

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._model.vocab_info


@dataclass(kw_only=True)
class QwenTokenizerConfig:
    use_im_end: bool = False


def _load_qwen_tokenizer(
    resolver: DependencyResolver, path: Path, name: str, config: QwenTokenizerConfig
) -> Tokenizer:
    eos_token = "<|im_end|>" if config.use_im_end else "<|endoftext|>"

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
    except OSError as ex:
        raise InfraError(
            f"A system error has occurred while reading the '{path}' tokenizer model. See the nested exception for details."
        ) from ex
    except RuntimeError as ex:
        raise TokenizerLoadError(
            name, f"The '{path}' tokenizer model cannot be loaded. See the nested exception for details."  # fmt: skip
        ) from ex

    return QwenTokenizer(model, eos_token)
