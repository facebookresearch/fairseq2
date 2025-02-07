# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)


def resolve_text_tokenizer_reference(
    asset_store: AssetStore, card: AssetCard
) -> AssetCard:
    name = card.name

    while True:
        try:
            ref_name = card.field("tokenizer_ref").as_(str)
        except AssetCardFieldNotFoundError:
            break

        try:
            card = asset_store.retrieve_card(ref_name)
        except AssetCardNotFoundError:
            raise AssetCardError(
                name, f"The '{ref_name}' asset card referenced by the '{name}' text tokenizer cannot be found."  # fmt: skip
            ) from None

    return card
