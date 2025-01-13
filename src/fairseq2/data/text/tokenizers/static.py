# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import AssetCard
from fairseq2.context import get_runtime_context
from fairseq2.data.text.tokenizers.handler import (
    TextTokenizerHandler,
    TextTokenizerNotFoundError,
    get_text_tokenizer_family,
)
from fairseq2.data.text.tokenizers.ref import resolve_text_tokenizer_reference
from fairseq2.data.text.tokenizers.tokenizer import TextTokenizer


def load_text_tokenizer(
    name_or_card: str | AssetCard, *, force: bool = False
) -> TextTokenizer:
    context = get_runtime_context()

    if isinstance(name_or_card, AssetCard):
        card = name_or_card
    else:
        card = context.asset_store.retrieve_card(name_or_card)

    card = resolve_text_tokenizer_reference(context.asset_store, card)

    family = get_text_tokenizer_family(card)

    registry = context.get_registry(TextTokenizerHandler)

    try:
        handler = registry.get(family)
    except LookupError:
        raise TextTokenizerNotFoundError(card.name) from None

    return handler.load(card, force=force)
