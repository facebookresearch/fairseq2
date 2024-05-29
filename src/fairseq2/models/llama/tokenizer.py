# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Final, Optional, final

from fairseq2.data.text import TiktokenEncoder, TiktokenTokenizer
from fairseq2.typing import Device, override


@final
class LLaMA3Tokenizer(TiktokenTokenizer):
    """Represents a LLaMA 3 tokenizer."""

    _SPLIT_REGEX: Final = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # fmt: skip

    _eos_token: str

    def __init__(self, path: Path, instruct: bool = False) -> None:
        """
        :param path:
            The path to the tiktoken BPE file.
        :param instruct:
            If ``True``, uses EOT (end-of-turn) token in-place of EOS token.
        """
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end-of-turn
        ]

        num_reserved_special_tokens = 256

        for i in range(5, num_reserved_special_tokens - 5):
            special_tokens.append(f"<|reserved_special_token_{i}|>")

        self._eos_token = "<|eot_id|>" if instruct else "<|end_of_text|>"

        super().__init__(
            path,
            split_regex=self._SPLIT_REGEX,
            unk_token=None,
            bos_token="<|begin_of_text|>",
            eos_token=self._eos_token,
            pad_token=None,
            special_tokens=special_tokens,
        )

    @override
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> TiktokenEncoder:
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")

        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")

        if mode is None or mode == "default":
            prefix_tokens = ["<|begin_of_text|>"]
            suffix_tokens = [self._eos_token]
        elif mode == "prompt":
            prefix_tokens = ["<|begin_of_text|>"]
            # In prompt mode, we expect the generator to finish the sequence.
            suffix_tokens = None
        elif mode == "prompt_response":
            prefix_tokens = []
            suffix_tokens = [self._eos_token]
        else:
            raise ValueError(
                f"`mode` must be 'default' or 'prompt', but is '{mode}' instead."
            )

        return TiktokenEncoder(
            self._encoding,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )
