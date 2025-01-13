# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import AssetCard, AssetCardFieldNotFoundError, AssetStore


def resolve_text_tokenizer_reference(
    asset_store: AssetStore, card: AssetCard
) -> AssetCard:
    while True:
        try:
            ref_name = card.field("tokenizer_ref").as_(str)
        except AssetCardFieldNotFoundError:
            break

        card = asset_store.retrieve_card(ref_name)

    return card
