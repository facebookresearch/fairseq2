# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma 4 tokenizer.

Reuses the same HuggingFace tokenizer infrastructure as Gemma 3n since the
Gemma 4 family uses the same tokenizer format and vocabulary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, final

from typing_extensions import override

from fairseq2.data.tokenizers import (
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    VocabularyInfo,
)
from fairseq2.data.tokenizers.hg import (
    HuggingFaceTokenDecoder,
    HuggingFaceTokenEncoder,
    HuggingFaceTokenModel,
    load_hg_token_model,
)
from fairseq2.device import Device

__all__ = ["Gemma4Tokenizer"]


@final
class Gemma4Tokenizer(Tokenizer):
    """Gemma 4 tokenizer wrapping HuggingFace ``tokenizer.json``."""

    def __init__(self, model: HuggingFaceTokenModel) -> None:
        self._model = model

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

        if mode is not None and mode not in ("default", "prompt", "as_is"):
            raise ValueError(
                f"`mode` must be 'default', 'prompt', or 'as_is', but is '{mode}' instead."
            )

        # Gemma 4 uses BOS token (ID 2) as prefix, no EOS suffix.
        if mode == "as_is":
            prefix_tokens: list[str] = []
            suffix_tokens: list[str] = []
        else:
            prefix_tokens = ["<bos>"]
            suffix_tokens = []

        return HuggingFaceTokenEncoder(
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

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Apply Gemma 4 chat template to format a conversation.

        :param conversation: List of messages with ``'role'`` and ``'content'``
            keys.  Roles can be ``'user'``, ``'assistant'`` / ``'model'``, or
            ``'system'``.
        :param tokenize: If ``True``, return token IDs.  If ``False``, return
            the formatted string.
        :param add_generation_prompt: If ``True``, append prompt for the model
            to continue.
        :param kwargs: Additional arguments passed to HuggingFace
            ``apply_chat_template``.
        :returns: Token IDs (``list[int]``) if *tokenize* is ``True``,
            formatted string otherwise.
        """
        return self._model._tok.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    @property
    def chat_template(self) -> str | None:
        """The current chat template (Jinja2 format), or ``None``."""
        return getattr(self._model._tok, "chat_template", None)


def load_gemma4_tokenizer(path: Path, config: None) -> Tokenizer:
    """Load Gemma 4 tokenizer from HuggingFace ``tokenizer.json``.

    :param path: Path to the tokenizer directory containing ``tokenizer.json``.
    :param config: Config (unused, always ``None`` for Gemma 4).
    :returns: A :class:`Gemma4Tokenizer` instance.
    """
    model = load_hg_token_model(
        path,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        boh_token=None,
        eoh_token=None,
    )

    # Use the chat template from the HF tokenizer as-is.  Gemma 4 uses
    # <|turn>/<turn|> markers (NOT the old <start_of_turn>/<end_of_turn>
    # format from Gemma 2/3).  The HF tokenizer_config.json already
    # contains the correct template with tool-call support.

    return Gemma4Tokenizer(model)
