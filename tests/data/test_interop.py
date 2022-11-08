# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.data import IString
from tests.common import TestCase


class TestString(TestCase):
    def test_len_returns_correct_length(self) -> None:
        s1 = "schöne Grüße!"
        s2 = IString("schöne Grüße!")

        self.assertEqual(len(s1), len(s2))

        # Grinning Face Emoji
        s1 = "\U0001f600"
        s2 = IString("\U0001f600")

        self.assertEqual(len(s1), len(s2))

        s1 = "Hello 🦆!"
        s2 = IString("Hello 🦆!")

        self.assertEqual(len(s1), len(s2))

    def test_len_returns_zero_if_string_is_empty(self) -> None:
        s = IString()

        self.assertEqual(len(s), 0)

        s = IString("")

        self.assertEqual(len(s), 0)

    def test_eq_returns_true_if_strings_are_equal(self) -> None:
        s1 = IString("schöne Grüße!")
        s2 = IString("schöne Grüße!")

        r = s1 == s2

        self.assertTrue(r)

        r = s1 != s2

        self.assertFalse(r)

    def test_eq_returns_true_if_string_and_python_string_are_equal(self) -> None:
        s1 = "schöne Grüße!"
        s2 = IString("schöne Grüße!")

        r = s1 == s2  # type: ignore[comparison-overlap]

        self.assertTrue(r)

        r = s2 == s1  # type: ignore[comparison-overlap]

        self.assertTrue(r)

        r = s1 != s2  # type: ignore[comparison-overlap]

        self.assertFalse(r)

        r = s2 != s1  # type: ignore[comparison-overlap]

        self.assertFalse(r)

    def test_eq_returns_false_if_strings_are_not_equal(self) -> None:
        s1 = IString("schöne Grüße!")
        s2 = IString("schone Grüße!")

        r = s1 == s2

        self.assertFalse(r)

        r = s1 != s2

        self.assertTrue(r)

    def test_eq_returns_false_if_string_and_python_string_are_not_equal(self) -> None:
        s1 = "schöne Grüße!"
        s2 = IString("schöne Grüsse!")

        r = s1 == s2  # type: ignore[comparison-overlap]

        self.assertFalse(r)

        r = s2 == s1  # type: ignore[comparison-overlap]

        self.assertFalse(r)

        r = s1 != s2  # type: ignore[comparison-overlap]

        self.assertTrue(r)

        r = s2 != s1  # type: ignore[comparison-overlap]

        self.assertTrue(r)

    def test_init_initializes_correctly_with_python_string(self) -> None:
        s1 = "schöne Grüße!"
        s2 = IString(s1)

        self.assertEqual(s1, s2)

    def test_to_py_returns_python_str(self) -> None:
        s = IString("schöne Grüße!")

        r = s.to_py()

        self.assertIsInstance(r, str)

        self.assertNotIsInstance(r, IString)

        self.assertEqual(r, "schöne Grüße!")

    def test_hash_returns_same_value_with_each_call(self) -> None:
        s = IString("schöne Grüsse!")

        h1 = hash(s)
        h2 = hash(s)

        self.assertEqual(h1, h2)

    def test_repr_returns_quoted_string(self) -> None:
        s1 = "schöne Grüße!"
        s2 = IString("schöne Grüße!")

        self.assertEqual(repr(s1), repr(s2))
