# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.context import RuntimeContext
from fairseq2.data.text.tokenizers import TextTokenizerHandler
from fairseq2.data.text.tokenizers.char_tokenizer import CharTokenizerHandler
from fairseq2.data.text.tokenizers.llama import LLaMATokenizerHandler
from fairseq2.data.text.tokenizers.mistral import MistralTokenizerHandler
from fairseq2.data.text.tokenizers.nllb import NllbTokenizerHandler
from fairseq2.data.text.tokenizers.s2t_transformer import S2TTransformerTokenizerHandler


def _register_text_tokenizers(context: RuntimeContext) -> None:
    asset_download_manager = context.asset_download_manager

    registry = context.get_registry(TextTokenizerHandler)

    handler: TextTokenizerHandler

    # Char Tokenizer
    handler = CharTokenizerHandler(asset_download_manager)

    registry.register(handler.family, handler)

    # LLaMA
    handler = LLaMATokenizerHandler(asset_download_manager)

    registry.register(handler.family, handler)

    # Mistral
    handler = MistralTokenizerHandler(asset_download_manager)

    registry.register(handler.family, handler)

    # NLLB
    handler = NllbTokenizerHandler(asset_download_manager)

    registry.register(handler.family, handler)

    # S2T Transformer
    handler = S2TTransformerTokenizerHandler(asset_download_manager)

    registry.register(handler.family, handler)
