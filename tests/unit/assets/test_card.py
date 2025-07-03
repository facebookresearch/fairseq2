# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import pytest

from fairseq2.assets import AssetCard, AssetCardError, AssetCardFieldNotFoundError


class TestAssetCard:
    def setup_method(self) -> None:
        root_metadata = {
            "name": "root-card",
            "field3": 3,
        }

        root_card = AssetCard("root-card", root_metadata)

        base_metadata: dict[str, object] = {
            "name": "base-card",
            "field1": "base-foo1",
            "field2": {"sub-field1": "sub-foo1"},
            "field8": [1, "b", 3],
        }

        base_card = AssetCard("base-card", base_metadata, root_card)

        metadata: dict[str, object] = {
            "name": "test-card",
            "field1": "foo1",
            "field2": {
                "sub-field2": "sub-foo2",
            },
            "field4": "",
            "field5": [],
            "field6": "invalid/filename",
            "field7": [1, 3, 2],
            "field9": "http://foo.com",
            "field10": None,
        }

        self.card = AssetCard("test-card", metadata, base_card)

    def test_field_works(self) -> None:
        value = self.card.field("field1").as_(str)

        assert value == "foo1"

        value = self.card.field("field2").field("sub-field1").as_(str)

        assert value == "sub-foo1"

        value = self.card.field("field2").field("sub-field2").as_(str)

        assert value == "sub-foo2"

        int_value = self.card.field("field3").as_(int)

        assert int_value == 3

        none_value = self.card.field("field10").as_(str | None)

        assert none_value is None

    def test_as_raises_error_when_field_type_is_incorrect(self) -> None:
        with pytest.raises(
            AssetCardError, match=rf"^The value of the 'field1' field of the 'test-card' asset card cannot be parsed as `{int}`. See the nested exception for details\.$",  # fmt: skip
        ):
            self.card.field("field1").as_(int)

        with pytest.raises(
            AssetCardError, match=rf"^The value of the 'field2\.sub-field1' field of the 'test-card' asset card cannot be parsed as `{int}`. See the nested exception for details\.$",  # fmt: skip
        ):
            self.card.field("field2").field("sub-field1").as_(int)

    def test_as_raises_error_when_field_does_not_exist(self) -> None:
        with pytest.raises(
            AssetCardFieldNotFoundError, match=r"^The 'test-card' asset card does not have a field named 'field11'\.$",  # fmt: skip
        ):
            self.card.field("field11").as_(str)

        with pytest.raises(
            AssetCardFieldNotFoundError, match=r"^The 'test-card' asset card does not have a field named 'field10\.sub-field'\.$",  # fmt: skip
        ):
            self.card.field("field10").field("sub-field").as_(str)

    def test_as_raises_error_when_field_is_empty(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"^The value of the 'field4' field of the 'test-card' asset card is empty\.$",
        ):
            self.card.field("field4").as_(str)

        with pytest.raises(
            AssetCardError, match=r"^The value of the 'field5' field of the 'test-card' asset card is empty\.$",  # fmt: skip
        ):
            self.card.field("field5").as_(list[str])

    def test_as_works_when_allow_empty_is_true(self) -> None:
        value1 = self.card.field("field4").as_(str, allow_empty=True)

        assert value1 == ""

        value2 = self.card.field("field5").as_(list[str], allow_empty=True)

        assert value2 == []

    def test_as_list_works(self) -> None:
        value = self.card.field("field7").as_(list[int])

        assert value == [1, 3, 2]

    def test_as_list_raises_error_when_field_is_not_a_valid_list(self) -> None:
        with pytest.raises(
            AssetCardError, match=r"The value of the 'field7' field of the 'test-card' asset card cannot be parsed as `list\[str\]`. See the nested exception for details\.$",  # fmt: skip
        ):
            self.card.field("field7").as_(list[str])

    def test_as_dict_works(self) -> None:
        value = self.card.field("field2").as_(dict[str, str])

        assert value == {"sub-field2": "sub-foo2"}

    def test_as_dict_raises_error_when_field_is_not_a_valid_dict(self) -> None:
        with pytest.raises(
            AssetCardError, match=r"The value of the 'field2' field of the 'test-card' asset card cannot be parsed as `dict\[str, int\]`. See the nested exception for details\.$",  # fmt: skip
        ):
            self.card.field("field2").as_(dict[str, int])

    def test_as_set_works(self) -> None:
        value = self.card.field("field7").as_(set[int])

        assert value == {1, 2, 3}

    def test_as_set_raises_error_when_field_is_not_a_valid_set(self) -> None:
        with pytest.raises(
            AssetCardError, match=r"The value of the 'field7' field of the 'test-card' asset card cannot be parsed as `set\[str\]`. See the nested exception for details\.$",  # fmt: skip
        ):
            self.card.field("field7").as_(set[str])

    def test_as_one_of_works(self) -> None:
        value = self.card.field("field1").as_one_of({"foo2", "foo1"})

        assert value == "foo1"

    def test_as_one_of_raises_error_when_field_is_not_one_of_valid_values(self) -> None:
        with pytest.raises(
            AssetCardError, match=r"The value of the 'field1' field of the 'test-card' asset card is expected to be one of the following values, but is 'foo1' instead: foo2, foo3$",  # fmt: skip
        ):
            self.card.field("field1").as_one_of({"foo3", "foo2"})

    def test_as_uri_works(self) -> None:
        self.card.field("field1").set("/foo1/foo2/")

        value = self.card.field("field1").as_uri()

        assert value == "file:///foo1/foo2"

        value = self.card.field("field9").as_uri()

        assert value == "http://foo.com"

        self.card.field("field1").set(Path("/foo1/foo2/"))

        value = self.card.field("field1").as_uri()

        assert value == "file:///foo1/foo2"

    def test_as_uri_raises_error_when_field_type_is_incorrect(self) -> None:
        with pytest.raises(
            AssetCardError, match=rf"The value of the 'field3' field of the 'test-card' asset card cannot be parsed as `{str}`. See the nested exception for details\.$",  # fmt: skip
        ):
            self.card.field("field3").as_uri()

    def test_as_uri_raises_error_when_field_is_not_uri_or_absolute_path(self) -> None:
        with pytest.raises(
            AssetCardError, match=r"The value of the 'field1' field of the 'test-card' asset card is a relative pathname \('foo1'\) and cannot be converted to a URI\.$",  # fmt: skip
        ):
            self.card.field("field1").as_uri()

    def test_set_works(self) -> None:
        self.card.field("field1").set("xyz")

        value = self.card.field("field1").as_(str)

        assert value == "xyz"

        self.card.field("field2").field("sub-field3").field("field").set("foo3")

        value = self.card.field("field2").field("sub-field3").field("field").as_(str)

        assert value == "foo3"

    def test_set_raises_error_when_path_conflicts(self) -> None:
        with pytest.raises(
            AssetCardError, match=r"^The 'test-card' asset card cannot have a field named 'field1.field2' due to path conflict at 'field1'\.$",  # fmt: skip
        ):
            self.card.field("field1").field("field2").set("foo")
