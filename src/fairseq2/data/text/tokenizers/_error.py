# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


class UnknownTextTokenizerError(Exception):
    tokenizer_name: str

    def __init__(self, tokenizer_name: str) -> None:
        super().__init__(f"'{tokenizer_name}' is not a known text tokenizer.")

        self.tokenizer_name = tokenizer_name


class UnknownTextTokenizerFamilyError(Exception):
    family: str
    tokenizer_name: str

    def __init__(self, family: str, tokenizer_name: str) -> None:
        super().__init__(
            f"The '{tokenizer_name}' text tokenizer has an unknown family '{family}'"
        )

        self.family = family
        self.tokenizer_name = tokenizer_name


class TextTokenizerLoadError(Exception):
    tokenizer_name: str

    def __init__(self, tokenizer_name: str, message: str) -> None:
        super().__init__(message)

        self.tokenizer_name = tokenizer_name


def text_tokenizer_asset_card_error(tokenizer_name: str) -> TextTokenizerLoadError:
    return TextTokenizerLoadError(
        tokenizer_name, f"The '{tokenizer_name}' asset card cannot be read. See the nested exception for details."  # fmt: skip
    )
