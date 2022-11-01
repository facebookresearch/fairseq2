import unittest
from typing import NoReturn

import hamcrest

from tests import tensor_matchers as tm


def throw(exception: Exception) -> NoReturn:
    """Throw an exception, useful inside a lambad.

    :param exception: the exception.
    """
    raise exception


class AssertMatchTest(unittest.TestCase):
    def test(self) -> None:
        tm.assert_match("abc", "abc")

        hamcrest.assert_that(
            lambda: tm.assert_match("abc", "xyz"),  # type: ignore
            hamcrest.raises(
                AssertionError,
                "Expected: 'xyz'",
            ),
        )


class AssertTruthyTest(unittest.TestCase):
    def test(self) -> None:
        tm.assert_true(True)
        tm.assert_true("abc")
        tm.assert_true(1)
        tm.assert_true([1])

        hamcrest.assert_that(
            lambda: tm.assert_true(False),  # type: ignore
            hamcrest.raises(
                AssertionError,
            ),
        )

        hamcrest.assert_that(
            lambda: tm.assert_true("", reason="meh"),  # type: ignore
            hamcrest.raises(
                AssertionError,
                "meh",
            ),
        )


class AssertFalseyTest(unittest.TestCase):
    def test(self) -> None:
        tm.assert_false(False)
        tm.assert_false("")
        tm.assert_false(0)
        tm.assert_false([])

        hamcrest.assert_that(
            lambda: tm.assert_false(True),  # type: ignore
            hamcrest.raises(
                AssertionError,
            ),
        )

        hamcrest.assert_that(
            lambda: tm.assert_false("abc", reason="meh"),  # type: ignore
            hamcrest.raises(
                AssertionError,
                "meh",
            ),
        )


class AssertRaisesTest(unittest.TestCase):
    def test_simple(self) -> None:
        tm.assert_raises(
            lambda: throw(ValueError("abc")),
            ValueError,
        )

        tm.assert_raises(lambda: throw(ValueError("abc")), ValueError, "abc")

        # No exception.
        hamcrest.assert_that(
            lambda: tm.assert_raises(  # type: ignore
                lambda: (),
                ValueError,
            ),
            hamcrest.raises(
                AssertionError,
                "No exception raised",
            ),
        )

        # Wrong exception type.
        hamcrest.assert_that(
            lambda: tm.assert_raises(  # type: ignore
                lambda: throw(ValueError("abc 123")), IndexError, "abc [0-9]+"
            ),
            hamcrest.raises(
                AssertionError,
                "was raised instead",
            ),
        )

    def test_regex(self) -> None:
        tm.assert_raises(lambda: throw(ValueError("abc 123")), ValueError, "abc [0-9]+")

        hamcrest.assert_that(
            lambda: tm.assert_raises(  # type: ignore
                lambda: throw(ValueError("abc xyz")), ValueError, "abc [0-9]+"
            ),
            hamcrest.raises(
                AssertionError,
                "the expected pattern .* not found",
            ),
        )

    def test_matching(self) -> None:
        class ExampleException(ValueError):
            code: int

        e = ExampleException("abc 123")
        e.code = 123

        tm.assert_raises(
            lambda: throw(e),  # type: ignore
            ValueError,
            matching=hamcrest.has_properties(code=123),
        )

        hamcrest.assert_that(
            lambda: tm.assert_raises(
                lambda: throw(e),  # type: ignore
                ValueError,
                matching=hamcrest.has_properties(code=9),
            ),
            hamcrest.raises(
                AssertionError,
                "Correct assertion type .* but an object with .* not found",
            ),
        )
