# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data.text.converters import StrSplitter


class TestStrSplitter:
    def test_call_works(self) -> None:
        s = "23\t9\t12\t\tabc\t34\t~~\t\t90\t 1 \t "

        splitter = StrSplitter()

        assert splitter(s) == [
            "23",
            "9",
            "12",
            "",
            "abc",
            "34",
            "~~",
            "",
            "90",
            " 1 ",
            " ",
        ]

    def test_call_works_when_separator_is_specified(self) -> None:
        s = "23 9 12  abc 34 ~~  90 \t 1  "

        splitter = StrSplitter(sep=" ")

        assert splitter(s) == [
            "23",
            "9",
            "12",
            "",
            "abc",
            "34",
            "~~",
            "",
            "90",
            "\t",
            "1",
            "",
            "",
        ]

    def test_call_works_when_input_is_empty(self) -> None:
        splitter = StrSplitter()

        assert splitter("") == [""]

    def test_call_works_when_names_are_specified(self) -> None:
        s = "1\t2\t3"

        splitter = StrSplitter(names=["a", "b", "c"])

        assert splitter(s) == {"a": "1", "b": "2", "c": "3"}

    def test_call_raises_error_when_fields_and_names_do_not_match(self) -> None:
        s = "1\t2\t3"

        splitter = StrSplitter(names=["a", "b"])

        with pytest.raises(
            ValueError,
            match=r"^The number of fields must match the number of names \(2\), but is 3 instead\.$",
        ):
            splitter(s)
