# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from os import PathLike
from pathlib import Path

import pytest

from fairseq2.assets import AssetCard, AssetCardError, AssetCardFieldNotFoundError


class TestAssetCard:
    def setup_method(self) -> None:
        root_metadata = {
            "name": "root-card",
            "field3": 3,
        }

        root_card = AssetCard(root_metadata)

        base_metadata = {
            "name": "base-card",
            "field1": "base-foo1",
            "field2": {"sub-field1": "sub-foo1"},
            "field8": [1, "b", 3],
        }

        base_card = AssetCard(base_metadata, root_card)

        metadata = {
            "name": "test-card",
            "field1": "foo1",
            "field2": {
                "sub-field2": "sub-foo2",
            },
            "field4": "",
            "field5": [],
            "field6": "invalid/filename",
            "field7": [1, 2, 3, 2],
            "field9": "http://foo.com",
        }

        self.card = AssetCard(metadata, base_card)

    def test_field_works(self) -> None:
        value = self.card.field("field1").as_(str)

        assert value == "foo1"

        value = self.card.field("field2").field("sub-field1").as_(str)

        assert value == "sub-foo1"

        value = self.card.field("field2").field("sub-field2").as_(str)

        assert value == "sub-foo2"

        int_value = self.card.field("field3").as_(int)

        assert int_value == 3

    def test_is_none_works(self) -> None:
        self.card.field("field7").set(None)

        assert self.card.field("field7").is_none()

        self.card.field("field2").field("sub_field1").set(None)

        assert self.card.field("field2").field("sub_field1").is_none()

    def test_is_none_raises_error_when_field_does_not_exist(self) -> None:
        self.card.field("field2").set(None)

        with pytest.raises(
            AssetCardFieldNotFoundError,
            match=r"^The asset card 'test-card' must have a field named 'field2.sub_field2'\.$",
        ):
            self.card.field("field2").field("sub_field2").is_none()

    def test_as_raises_error_when_field_type_is_incorrect(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=rf"^The value of the field 'field1' of the asset card 'test-card' must be of type `{int}`, but is of type `{str}` instead\.$",
        ):
            self.card.field("field1").as_(int)

        with pytest.raises(
            AssetCardError,
            match=rf"^The value of the field 'field2\.sub-field1' of the asset card 'test-card' must be of type `{int}`, but is of type `{str}` instead\.$",
        ):
            self.card.field("field2").field("sub-field1").as_(int)

    def test_as_raises_error_when_field_does_not_exist(self) -> None:
        with pytest.raises(
            AssetCardFieldNotFoundError,
            match=r"^The asset card 'test-card' must have a field named 'field10'\.$",
        ):
            self.card.field("field10").as_(str)

        with pytest.raises(
            AssetCardError,
            match=r"^The asset card 'test-card' must have a field named 'field10\.sub-field'\.$",
        ):
            self.card.field("field10").field("sub-field").as_(str)

    def test_as_raises_error_when_field_is_empty(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"^The value of the field 'field4' of the asset card 'test-card' must not be empty\.$",
        ):
            self.card.field("field4").as_(str)

        with pytest.raises(
            AssetCardError,
            match=r"^The value of the field 'field5' of the asset card 'test-card' must not be empty\.$",
        ):
            self.card.field("field5").as_list(str)

    def test_as_works_when_allow_empty_is_true(self) -> None:
        value1 = self.card.field("field4").as_(str, allow_empty=True)

        assert value1 == ""

        value2 = self.card.field("field5").as_list(str, allow_empty=True)

        assert value2 == []

    def test_as_list_works(self) -> None:
        value = self.card.field("field7").as_list(int)

        assert value == [1, 2, 3, 2]

    def test_as_list_raises_error_when_field_is_not_a_valid_list(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=rf"The elements of the field 'field7' of the asset card 'test-card' must be of type `{str}`, but the element at index 0 is of type `{int}` instead\.$",
        ):
            self.card.field("field7").as_list(str)

    def test_as_dict_works(self) -> None:
        value = self.card.field("field2").as_dict(str)

        assert value == {"sub-field2": "sub-foo2"}

    def test_as_dict_raises_error_when_field_is_not_a_valid_dict(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=rf"The items of the field 'field2' of the asset card 'test-card' must be of type `{int}`, but the item 'sub-field2' is of type `{str}` instead\.$",
        ):
            self.card.field("field2").as_dict(int)

    def test_as_set_works(self) -> None:
        value = self.card.field("field7").as_set(int)

        assert value == {1, 2, 3}

    def test_as_set_raises_error_when_field_is_not_a_valid_set(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=rf"The elements of the field 'field7' of the asset card 'test-card' must be of type `{str}`, but the element at index 0 is of type `{int}` instead\.$",
        ):
            self.card.field("field7").as_set(str)

    def test_as_one_of_works(self) -> None:
        value = self.card.field("field1").as_one_of({"foo2", "foo1"})

        assert value == "foo1"

    def test_as_one_of_raises_error_when_field_is_not_one_of_valid_values(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=rf"The value of the field 'field1' of the asset card 'test-card' must be one of \{['foo2', 'foo3']}, but is 'foo1' instead\.$",
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
            AssetCardError,
            match=rf"The value of the field 'field3' of the asset card 'test-card' must be of type `{str}` or `{PathLike}`, but is of type `{int}` instead\.$",
        ):
            self.card.field("field3").as_uri()

    def test_as_uri_raises_error_when_field_is_not_uri_or_absolute_path(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"The value of the field 'field1' of the asset card 'test-card' must be a URI or an absolute pathname, but is 'foo1' instead\.$",
        ):
            self.card.field("field1").as_uri()

    def test_as_filename_works(self) -> None:
        value = self.card.field("field1").as_filename()

        assert value == "foo1"

    def test_as_filename_raises_error_when_field_is_not_filename(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"^The value of the field 'field6' of the asset card 'test-card' must be a filename, but is 'invalid/filename' instead\.$",
        ):
            self.card.field("field6").as_filename()

    def test_set_works(self) -> None:
        self.card.field("field1").set("xyz")

        value = self.card.field("field1").as_(str)

        assert value == "xyz"

        self.card.field("field2").field("sub-field3").field("field").set("foo3")

        value = self.card.field("field2").field("sub-field3").field("field").as_(str)

        assert value == "foo3"

    def test_set_raises_error_when_path_conflicts(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"^The asset card 'test-card' cannot have a field named 'field1.field2' due to path conflict at 'field1'\.$",
        ):
            self.card.field("field1").field("field2").set("foo")

    def test_check_equals_works(self) -> None:
        self.card.field("field1").check_equals("foo1")

    def test_check_equals_raises_error_when_field_is_not_equal(self) -> None:
        with pytest.raises(
            AssetCardError,
            match=r"The value of the field 'field1' of the asset card 'test-card' must be 'foo2', but is 'foo1' instead\.$",
        ):
            self.card.field("field1").check_equals("foo2")
