# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from fairseq2.data import CString


class TestCString:
    def test_len_returns_correct_length(self) -> None:
        s1 = "schöne Grüße!"
        s2 = CString("schöne Grüße!")

        assert len(s1) == len(s2)

        # Grinning Face Emoji
        s1 = "\U0001f600"
        s2 = CString("\U0001f600")

        assert len(s1) == len(s2)

        s1 = "Hello 🦆!"
        s2 = CString("Hello 🦆!")

        assert len(s1) == len(s2)

    def test_len_returns_zero_if_string_is_empty(self) -> None:
        s = CString()

        assert len(s) == 0

        s = CString("")

        assert len(s) == 0

    def test_eq_returns_true_if_strings_are_equal(self) -> None:
        s1 = CString("schöne Grüße!")
        s2 = CString("schöne Grüße!")

        r = s1 == s2

        assert r

        r = s1 != s2

        assert not r

    def test_eq_returns_true_if_string_and_python_string_are_equal(self) -> None:
        s1 = "schöne Grüße!"
        s2 = CString("schöne Grüße!")

        r = s1 == s2  # type: ignore[comparison-overlap]

        assert r

        r = s2 == s1  # type: ignore[comparison-overlap]

        assert r

        r = s1 != s2  # type: ignore[comparison-overlap]

        assert not r

        r = s2 != s1  # type: ignore[comparison-overlap]

        assert not r

    def test_eq_returns_false_if_strings_are_not_equal(self) -> None:
        s1 = CString("schöne Grüße!")
        s2 = CString("schone Grüße!")

        r = s1 == s2

        assert not r

        r = s1 != s2

        assert r

    def test_eq_returns_false_if_string_and_python_string_are_not_equal(self) -> None:
        s1 = "schöne Grüße!"
        s2 = CString("schöne Grüsse!")

        r = s1 == s2  # type: ignore[comparison-overlap]

        assert not r

        r = s2 == s1  # type: ignore[comparison-overlap]

        assert not r

        r = s1 != s2  # type: ignore[comparison-overlap]

        assert r

        r = s2 != s1  # type: ignore[comparison-overlap]

        assert r

    def test_init_initializes_correctly_with_python_string(self) -> None:
        s1 = "schöne Grüße!"
        s2 = CString(s1)

        assert s1 == s2

    def test_hash_returns_same_value_with_each_call(self) -> None:
        s = CString("schöne Grüsse!")

        h1 = hash(s)
        h2 = hash(s)

        assert h1 == h2

    def test_str_returns_python_str(self) -> None:
        s = CString("schöne Grüße!")

        r = str(s)

        assert isinstance(r, str)

        assert not isinstance(r, CString)

        assert r == "schöne Grüße!"

    def test_repr_returns_quoted_string(self) -> None:
        s = CString("schöne Grüße!")

        assert "CString('schöne Grüße!')" == repr(s)

    def test_split_returns_correct_list(self) -> None:
        s = CString("hello world! this is a string")

        r = s.split(sep=" ")

        assert r == ["hello", "world!", "this", "is", "a", "string"]

        r = s.split("h")

        assert r == ["", "ello world! t", "is is a string"]

        r = s.split(sep="g")

        assert r == ["hello world! this is a strin", ""]

    def test_split_returns_correct_list_with_default_separator(self) -> None:
        s = CString("hello\tworld!\tthis\tis\ta\tstring")

        r = s.split()

        assert r == ["hello", "world!", "this", "is", "a", "string"]

    @pytest.mark.parametrize("v", ["", "  "])
    def test_split_returs_list_of_length_1_if_the_string_is_empty(self, v: str) -> None:
        s = CString(v)

        r = s.split()

        assert r == [v]

    def test_split_returns_empty_strings(self) -> None:
        s = CString(":  :: :")

        r = s.split(":")

        assert r == ["", "  ", "", " ", ""]

    def test_split_returns_list_of_length_1_if_separator_is_not_found(self) -> None:
        s = CString("hello world! this is a string")

        r = s.split(sep="z")

        assert r == ["hello world! this is a string"]

    def test_split_raises_error_if_separater_is_not_char(self) -> None:
        s = CString("hello world! this is a string")

        with pytest.raises(ValueError, match=r"^`sep` must be of length 1\.$"):
            s.split("<>")
