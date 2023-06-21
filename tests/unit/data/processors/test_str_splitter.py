# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data.processors import StrSplitter


class TestStrToTensorConverter:
    def test_splits_as_expected(self) -> None:
        s = "23\t9\t12\t\tabc\t34\t~~\t\t90\t 1 \t "

        splitter = StrSplitter()

        assert splitter(s) == ["23", "9", "12", "abc", "34", "~~", "90", " 1 ", " "]

    def test_splits_as_expected_with_custom_sep(self) -> None:
        s = "23 9 12  abc 34 ~~  90 \t 1  "

        splitter = StrSplitter(sep=" ")

        assert splitter(s) == ["23", "9", "12", "abc", "34", "~~", "90", "\t", "1"]

    @pytest.mark.parametrize("s", ["", "\t\t", "\t\t\t"])
    def test_splits_empty_string_as_expected(self, s: str) -> None:
        splitter = StrSplitter()

        assert splitter(s) == []
