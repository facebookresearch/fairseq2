# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import (
    AssetCard,
    default_asset_download_manager,
    default_asset_store,
)
from fairseq2.data.text.tokenizers.ref import resolve_text_tokenizer_reference
from fairseq2.data.text.tokenizers.registry import (
    TextTokenizerRegistry,
    get_text_tokenizer_family,
)
from fairseq2.data.text.tokenizers.tokenizer import TextTokenizer

default_text_tokenizer_registry = TextTokenizerRegistry()


def load_text_tokenizer(
    name_or_card: str | AssetCard, *, force: bool = False
) -> TextTokenizer:
    if isinstance(name_or_card, AssetCard):
        card = name_or_card
    else:
        card = default_asset_store.retrieve_card(name_or_card)

    card = resolve_text_tokenizer_reference(default_asset_store, card)

    family = get_text_tokenizer_family(card)

    handler = default_text_tokenizer_registry.get(family)

    return handler.load(card, default_asset_download_manager, force=force)
