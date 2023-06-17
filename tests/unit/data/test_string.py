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

    def test_split_str(self) -> None:
        s = CString("Hello world! This is a string")
        expected = ["Hello", "world!", "This", "is", "a", "string"]

        assert expected == s.split(sep=" ")

    def test_split_str_edge_cases(self) -> None:
        s = CString("hello world! this is a string")

        # split sep at start
        expected = ["ello world! t", "is is a string"]
        assert expected == s.split(sep="h")

        # split sep at end
        expected = ["hello world! this is a strin"]
        assert expected == s.split(sep="g")

        # split sep not in string
        expected = ["hello world! this is a string"]
        assert expected == s.split(sep="z")

    def test_split_str_consecutive_sep(self) -> None:
        s = CString("hello world!  this is    a string")

        expected = ["hello", "world!", "this", "is", "a", "string"]
        actual = s.split(sep=" ")
        assert expected == actual

    def test_split_str_invalid_argument(self) -> None:
        s = CString("Hello world! This is a string")

        s.split("l")
        with pytest.raises(ValueError):
            s.split()
        with pytest.raises(ValueError):
            s.split("<>")
