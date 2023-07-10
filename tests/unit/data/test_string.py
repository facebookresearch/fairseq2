# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import pytest

from fairseq2.data import CString


class TestCString:
    def test_init_works_when_input_is_a_python_string(self) -> None:
        s1 = "schöne Grüße!"
        s2 = CString(s1)

        assert s1 == s2

    def test_len_works(self) -> None:
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

    def test_len_works_when_string_is_empty(self) -> None:
        s = CString()

        assert len(s) == 0

        s = CString("")

        assert len(s) == 0

    def test_eq_works_when_strings_are_equal(self) -> None:
        s1 = CString("schöne Grüße!")
        s2 = CString("schöne Grüße!")

        r = s1 == s2

        assert r

        r = s1 != s2

        assert not r

    def test_eq_works_when_string_and_python_string_are_equal(self) -> None:
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

    def test_eq_works_when_strings_are_not_equal(self) -> None:
        s1 = CString("schöne Grüße!")
        s2 = CString("schone Grüße!")

        r = s1 == s2

        assert not r

        r = s1 != s2

        assert r

    def test_eq_works_when_string_and_python_string_are_not_equal(self) -> None:
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

    def test_hash_works_when_called_multiple_times(self) -> None:
        s = CString("schöne Grüsse!")

        h1 = hash(s)
        h2 = hash(s)

        assert h1 == h2

    def test_str_works(self) -> None:
        s = CString("schöne Grüße!")

        r = str(s)

        assert isinstance(r, str)

        assert not isinstance(r, CString)

        assert r == "schöne Grüße!"

    def test_repr_works(self) -> None:
        s = CString("schöne Grüße!")

        assert "CString('schöne Grüße!')" == repr(s)

    def test_split_works_when_no_separator_is_specified(self) -> None:
        s = CString("hello\tworld!\tthis\tis\ta\tstring")

        r = s.split()

        assert r == ["hello", "world!", "this", "is", "a", "string"]

    def test_split_works_when_separator_is_specified(self) -> None:
        s = CString("hello world! this is a string")

        r = s.split(sep=" ")

        assert r == ["hello", "world!", "this", "is", "a", "string"]

        r = s.split("h")

        assert r == ["", "ello world! t", "is is a string"]

        r = s.split(sep="g")

        assert r == ["hello world! this is a strin", ""]

    @pytest.mark.parametrize("v", ["", "  "])
    def test_split_works_when_string_is_empty(self, v: str) -> None:
        s = CString(v)

        r = s.split()

        assert r == [v]

    def test_split_works_when_string_contains_only_separators(self) -> None:
        s = CString(":  :: :")

        r = s.split(":")

        assert r == ["", "  ", "", " ", ""]

    def test_split_works_when_separator_is_not_found_in_string(self) -> None:
        s = CString("hello world! this is a string")

        r = s.split(sep="z")

        assert r == ["hello world! this is a string"]

    def test_split_raises_error_when_separator_is_not_char(self) -> None:
        s = CString("hello world! this is a string")

        with pytest.raises(
            ValueError,
            match=r"^`sep` must be of length 1, but is of length 2 instead\.$",
        ):
            s.split("<>")

    def test_pickle_works(self) -> None:
        s = CString("hello world!")

        dump = pickle.dumps(s)

        del s

        s = pickle.loads(dump)

        assert s == "hello world!"
