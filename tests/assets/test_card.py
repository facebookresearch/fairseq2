# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.assets import AssetCard, AssetCardError, AssetCardFieldNotFoundError


class TestAssetCard:
    def setup_method(self) -> None:
        root_data = {
            "field3": 3,
        }

        root_card = AssetCard("root-card", root_data)

        base_data = {
            "field1": "base-foo1",
            "field8": [1, "b", 3],
        }

        base_card = AssetCard("base-card", base_data, root_card)

        data = {
            "field1": "foo1",
            "field2": {
                "sub-field": "sub-foo",
            },
            "field4": "",
            "field5": [],
            "field6": "invalid/filename",
            "field7": [1, 2, 3],
            "field9": "http://foo.com",
        }

        self.card = AssetCard("test-card", data, base_card)

    def test_returns_correct_field_value(self) -> None:
        value = self.card.field("field1").as_(str)

        assert value == "foo1"

    def test_returns_correct_base_field_value(self) -> None:
        value = self.card.field("field3").as_(int)

        assert value == 3

    def test_returns_correct_sub_field_value(self) -> None:
        value = self.card.field("field2").field("sub-field").as_(str)

        assert value == "sub-foo"

    def test_raises_error_if_field_type_is_incorrect(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"^The type of the field 'field1' of the asset card 'test-card' must be <class 'int'>, but is <class 'str'> instead\.$",
        ):
            self.card.field("field1").as_(int)

    def test_raises_error_if_sub_field_type_is_incorrect(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"^The type of the field 'field2\.sub-field' of the asset card 'test-card' must be <class 'int'>, but is <class 'str'> instead\.$",
        ):
            self.card.field("field2").field("sub-field").as_(int)

    def test_raises_error_if_field_does_not_exist(self) -> None:
        with pytest.raises(
            AssetCardFieldNotFoundError,
            match=r"^The asset card 'test-card' must have a field named 'field10'\.$",
        ):
            self.card.field("field10").as_(str)

    def test_raises_error_if_sub_field_does_not_exist(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"^The asset card 'test-card' must have a field named 'field10\.sub-field'\.$",
        ):
            self.card.field("field10").field("sub-field").as_(str)

    def test_raises_error_if_field_value_is_empty(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"^The value of the field 'field4' of the asset card 'test-card' must be non-empty\.$",
        ):
            self.card.field("field4").as_(str)

        with pytest.raises(
            AssetCardError,
            match=r"^The value of the field 'field5' of the asset card 'test-card' must be non-empty\.$",
        ):
            self.card.field("field5").as_list(str)

    def test_returns_empty_if_empty_field_value_is_allowed(self) -> None:
        value1 = self.card.field("field4").as_(str, allow_empty=True)

        assert value1 == ""

        value2 = self.card.field("field5").as_list(str, allow_empty=True)

        assert value2 == []

    def test_returns_field_value_if_it_is_valid_url(self) -> None:
        value = self.card.field("field9").as_uri()

        assert value == "http://foo.com"

    def test_raises_error_if_field_value_is_not_a_valid_uri(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"The value of the field 'field1' of the asset card 'test-card' must be a valid URI, but is 'foo1' instead\.$",
        ):
            self.card.field("field1").as_uri()

    def test_returns_field_value_if_it_is_valid_filename(self) -> None:
        value = self.card.field("field1").as_filename()

        assert value == "foo1"

    def test_raises_error_if_field_value_is_not_a_valid_filename(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"^The value of the field 'field6' of the asset card 'test-card' must be a valid filename, but is 'invalid/filename' instead\.$",
        ):
            self.card.field("field6").as_filename()

    def test_returns_field_value_if_it_is_valid_list(self) -> None:
        value = self.card.field("field7").as_list(int)

        assert value == [1, 2, 3]

    def test_raises_error_if_field_value_is_not_a_valid_list(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"The value of the field 'field8' of the asset card 'test-card' must be a list of type of <class 'int'>, but has at least one element of type <class 'str'>\.$",
        ):
            self.card.field("field8").as_list(int)

    def test_returns_field_value_if_it_is_one_of_valid_values(self) -> None:
        value = self.card.field("field1").as_one_of({"foo2", "foo1"})

        assert value == "foo1"

    def test_raises_error_if_field_value_is_not_one_of_valid_values(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"The value of the field 'field1' of the asset card 'test-card' must be one of \['foo2', 'foo3'\], but is 'foo1' instead\.$",
        ):
            self.card.field("field1").as_one_of({"foo2", "foo3"})

    def test_check_equals_raises_no_error_if_equal(self) -> None:
        self.card.field("field1").check_equals("foo1")

    def test_check_equals_raises_error_if_not_equal(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"The value of the field 'field1' of the asset card 'test-card' must be 'foo2', but is 'foo1' instead\.$",
        ):
            self.card.field("field1").check_equals("foo2")
