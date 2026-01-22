# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from fairseq2.assets import AssetCard, AssetCardError


class TestAssetCard:
    def setup_method(self) -> None:
        metadata: dict[str, object]

        metadata = {"field2": 3}

        card = AssetCard("root-card", metadata)

        metadata = {"field1": "xyz", "field3": None}

        card = AssetCard("base-card", metadata, card)

        metadata = {
            "field1": "foo1",
            "field4": "/foo1/foo2",
            "field5": "http://foo.com",
        }

        self.card = AssetCard("test-card", metadata, card)

    def test_field_works(self) -> None:
        value = self.card.field("field1").value

        assert value == "foo1"

        value = self.card.field("field2").value

        assert value == 3

        value = self.card.field("field3").value

        assert value is None

    def test_field_raises_error_when_field_does_not_exist(self) -> None:
        with pytest.raises(
            AssetCardError, match=r"^test-card asset card does not have a field named field9\.$",  # fmt: skip
        ):
            self.card.field("field9").value

    def test_as_raises_error_when_field_type_is_incorrect(self) -> None:
        with pytest.raises(
            AssetCardError, match=rf"field2 field of the test-card asset card is expected to be of type `{bool}`, but is of type `{int}` instead\.$",  # fmt: skip
        ):
            self.card.field("field2").as_(bool)

    def test_as_uri_works(self) -> None:
        value = self.card.field("field4").as_uri()

        assert str(value) == "file:///foo1/foo2"

        value = self.card.field("field5").as_uri()

        assert str(value) == "http://foo.com"

    def test_as_uri_raises_error_when_field_is_not_uri_or_absolute_path(self) -> None:
        with pytest.raises(
            AssetCardError, match=r"field1 field of the test-card asset card is a relative pathname and cannot be converted to a URI\.$",  # fmt: skip
        ):
            self.card.field("field1").as_uri()
