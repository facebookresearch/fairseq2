# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.assets import AssetCard, AssetCardError, AssetStore


def resolve_tokenizer_reference(asset_store: AssetStore, card: AssetCard) -> AssetCard:
    name = card.name

    while True:
        ref_field = card.maybe_get_field("tokenizer_ref")
        if ref_field is None:
            break

        ref_name = ref_field.as_(str)

        ref_card = asset_store.maybe_retrieve_card(ref_name)
        if ref_card is None:
            msg = f"{ref_name} asset referenced by the {name} asset card does not exist."  # fmt: skip

            raise AssetCardError(name, msg)

        card = ref_card

    return card
