import contextlib
import numbers
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import hamcrest
import torch
from hamcrest.core.assert_that import _assert_bool, _assert_match
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
from overrides import overrides

T = TypeVar("T")

# int is not a Number?
# https://github.com/python/mypy/issues/3186
# https://stackoverflow.com/questions/69334475/how-to-hint-at-number-types-i-e-subclasses-of-number-not-numbers-themselv/69383462#69383462kk

NumberLike = Union[numbers.Number, numbers.Complex, SupportsFloat]

TensorConvertable = Any

# TensorConvertable = Union[
#    torch.Tensor,
#    NumberLike,
#    Sequence,
#    List,
#    Tuple,
#    nptyping.NDArray,
# ]
"Types which torch.as_tensor(T) can convert."


def hide_module_tracebacks(module: Any, mode: bool = True) -> None:
    # unittest integration; hide these frames from tracebacks
    module["__unittest"] = mode
    # py.test integration; hide these frames from tracebacks
    module["__tracebackhide__"] = mode


def hide_tracebacks(mode: bool = True) -> None:
    """Hint that some unittest stacks (unittest, pytest) should remove frames
    from tracebacks that include this module.

    :param mode: optional, the traceback mode.
    """
    hide_module_tracebacks(globals(), mode)


hide_tracebacks(True)

hide_module_tracebacks(hamcrest.core.base_matcher.__dict__)


@dataclass
class WhenCalledMatcher(BaseMatcher[Callable[..., T]]):
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    matcher: Matcher[T]
    method: Optional[str] = None

    def __init__(
        self,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
        matcher: Matcher[T],
        method: Optional[str] = None,
    ) -> None:
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.matcher = matcher
        self.method = method

    def _matches(self, item: Callable[..., T]) -> bool:
        return self.matcher.matches(self._call_item(item))

    def _call_item(self, item: Callable[..., T]) -> Any:
        if self.method is None:
            f = item
        else:
            f = getattr(item, self.method)

        return f(*self.args, **self.kwargs)

    def describe_to(self, description: Description) -> None:
        call_sig = tuple(
            [repr(a) for a in self.args]
            + [f"{k}={repr(v)}" for k, v in self.kwargs.items()]
        )

        if self.method is None:
            f = "\\"
        else:
            f = "." + self.method

        description.append_text(f"{f}{call_sig}=>")

        description.append_description_of(self.matcher)

    def describe_mismatch(
        self,
        item: Callable[..., T],
        mismatch_description: Description,
    ) -> None:
        val = self._call_item(item)
        mismatch_description.append_text("was =>").append_description_of(val)


@dataclass
class WhenCalledBuilder:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    method: Optional[str] = None

    def matches(self, matcher: Union[T, Matcher[T]]) -> WhenCalledMatcher[T]:
        return WhenCalledMatcher(
            args=self.args,
            kwargs=self.kwargs,
            matcher=_as_matcher(matcher),
            method=self.method,
        )


def when_called(*args: Any, **kwargs: Any) -> WhenCalledBuilder:
    return WhenCalledBuilder(args, kwargs)


def calling_method(method: str, *args: Any, **kwargs: Any) -> WhenCalledBuilder:
    return WhenCalledBuilder(args, kwargs, method=method)


def _as_matcher(matcher: Union[T, Matcher[T]]) -> Matcher[T]:
    if isinstance(matcher, Matcher):
        return matcher

    return hamcrest.is_(matcher)


def assert_match(actual: Any, matcher: Any, reason: str = "") -> None:
    """Asserts that the actual value matches the matcher.

    Similar to hamcrest.assert_that(), but if the matcher is not a Matcher,
    will fallback to ``hamcrest.is_(matcher)`` rather than boolean matching.

    :param actual: the value to match.
    :param matcher: a matcher, or a value that will be converted to an ``is_`` matcher.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_match(
        actual=actual,
        matcher=_as_matcher(matcher),
        reason=reason,
    )


def assert_close_to(
    actual: Any,
    expected: Any,
    delta: Optional[SupportsFloat] = None,
    *,
    rtol: SupportsFloat = 1e-05,
    atol: SupportsFloat = 1e-08,
) -> None:
    """Asserts that two values are close to each other.

    :param actual: the actual value.
    :param expected: the expected value.
    :param delta: (optional) the tolerance.
    :param rtol: if delta is None, the relative tolerance.
    :param atol: if delta is None, the absolute tolerance.
    :return:
    """
    if delta is None:
        # numpy.isclose() pattern:
        delta = atol + rtol * abs(expected)
    assert_match(
        actual,
        hamcrest.close_to(expected, delta),  # type: ignore
    )


def assert_true(actual: Any, reason: str = "") -> None:
    """Asserts that the actual value is truthy.

    :param actual: the value to match.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_bool(actual, reason=reason)


def assert_false(actual: Any, reason: str = "") -> None:
    """Asserts that the actual value is falsey.

    :param actual: the value to match.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_bool(not actual, reason=reason)


def assert_raises(
    func: Callable[[], Any],
    exception: Type[Exception],
    pattern: Optional[str] = None,
    matching: Any = None,
) -> None:
    """Utility wrapper for ``hamcrest.assert_that(func,
    hamcrest.raises(...))``.

    :param func: the function to call.
    :param exception: the exception class to expect.
    :param pattern: an optional regex to match against the exception message.
    :param matching: optional Matchers to match against the exception.
    """

    hamcrest.assert_that(
        func,  # type: ignore
        hamcrest.raises(
            exception=exception,
            pattern=pattern,
            matching=matching,
        ),
    )


@contextlib.contextmanager
def ignore_warnings() -> Generator[None, None, None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def assert_tensor_views(*views: torch.Tensor) -> None:
    """Assert that each tensor is a view of the same storage..

    :param views: a series of child Tensors which must all be views of source.
    """
    if views:
        reference = views[0]
        views = views[1:]

    for t in views:
        assert_match(
            t.storage().data_ptr(),
            reference.storage().data_ptr(),
        )


def assert_tensor_storage_differs(
    tensor: torch.Tensor, reference: torch.Tensor
) -> None:
    """Assert that two tensors are not views of each other, and have different
    storage.

    :param tensor: the tensor.
    :param reference: the reference tensor.
    """
    assert_match(
        tensor.storage().data_ptr(),
        hamcrest.not_(reference.storage().data_ptr()),
    )


class TensorStructureMatcher(BaseMatcher[torch.Tensor]):
    """PyHamcrest matcher for comparing the structure of a tensor to an
    exemplar.

    Matches:
      - device
      - size
      - dtype
      - layout
    """

    expected: torch.Tensor

    def __init__(self, expected: TensorConvertable) -> None:
        self.expected = torch.as_tensor(expected)

    @overrides
    def _matches(self, item: torch.Tensor) -> bool:
        # Todo: structural miss-match that still shows expected tensor.

        try:
            assert_match(
                item.device,
                self.expected.device,
            )
            assert_match(
                item.size(),
                self.expected.size(),
            )
            assert_match(
                item.dtype,
                self.expected.dtype,
            )
            assert_match(
                item.layout,
                self.expected.layout,
            )
            return True
        except AssertionError:
            return False

    @overrides
    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.expected)


def matches_tensor_structure(
    expected: TensorConvertable,
) -> TensorStructureMatcher:
    """Construct a matcher for comparing the structure of a tensor to an
    exemplar.

    Matches on:
      - device
      - size
      - dtype
      - layout

    :return: a matcher.
    """
    return TensorStructureMatcher(expected)


def assert_tensor_structure(
    actual: torch.Tensor,
    expected: TensorConvertable,
) -> None:
    """Assert that the `actual` matches the structure (not data) of the
    `expected`.

    :param actual: a tensor.
    :param expected: an expected structure.
    """
    hamcrest.assert_that(
        actual,
        matches_tensor_structure(expected),
    )


class TensorMatcher(TensorStructureMatcher):
    """PyHamcrest matcher for comparing the structure and data a tensor to an
    exemplar.

    Matches:
      - device
      - size
      - dtype
      - layout
    """

    close: bool = False
    "Should <close> values be considered identical?"

    def __init__(
        self,
        expected: TensorConvertable,
        *,
        close: bool = False,
    ):
        super().__init__(expected=expected)
        self.close = close

        if self.expected.is_sparse and not self.expected.is_coalesced():
            self.expected = self.expected.coalesce()

    @overrides
    def _matches(self, item: Any) -> bool:
        if not super()._matches(item):
            return False

        if self.close:
            try:
                torch.testing.assert_close(
                    item,
                    self.expected,
                    equal_nan=True,
                )
                return True
            except AssertionError:
                return False

        else:
            if self.expected.is_sparse:
                assert_true(item.is_sparse)

                # TODO: it may be necessary to sort the indices and values.
                if not item.is_coalesced():
                    item = item.coalesce()

                assert_tensor_equals(
                    item.indices(),
                    self.expected.indices(),
                )
                assert_tensor_equals(
                    item.values(),
                    self.expected.values(),
                )
                return True
            else:
                # torch.equal(item, self.expected) does not support nan.
                try:
                    torch.testing.assert_close(
                        item,
                        self.expected,
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                except AssertionError:
                    return False
                return True

    @overrides
    def describe_to(self, description: Description) -> None:
        description.append_text("\n")
        description.append_description_of(self.expected)

    @overrides
    def describe_match(
        self,
        item: Any,
        match_description: Description,
    ) -> None:
        match_description.append_text("was \n")
        match_description.append_description_of(item)

    @overrides
    def describe_mismatch(
        self,
        item: Any,
        mismatch_description: Description,
    ) -> None:
        torch.set_printoptions(
            precision=10,
        )
        mismatch_description.append_text("was \n")
        mismatch_description.append_description_of(item)


def matches_tensor(
    expected: TensorConvertable,
    close: bool = False,
) -> TensorMatcher:
    """Returns a matcher for structure and value of a Tensor.

    :param expected: the expected Tensor.
    :param close: should *close* values be acceptable?
    """
    return TensorMatcher(expected, close=close)


def assert_tensor_equals(
    actual: torch.Tensor,
    expected: TensorConvertable,
    *,
    close: bool = False,
    view_of: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assert that the `actual` tensor equals the `expected` tensor.

    :param actual: the actual tensor.
    :param expected: the value (to coerce to a Tensor) to compare to.
    :param close: should *close* values match?
    :param view_of: if present, also check that actual is a view of the reference Tensor.
    :returns: the `actual` value.
    """
    hamcrest.assert_that(
        actual,
        matches_tensor(
            expected,
            close=close,
        ),
    )
    if view_of is not None:
        assert_tensor_views(view_of, actual)

    return actual


def match_tensor_sequence(
    *expected: TensorConvertable,
) -> Matcher[Sequence[torch.Tensor]]:
    """Returns a matcher which expects a sequence that matches the tensors.

    :param expected: the expected tensors.
    """
    return hamcrest.contains_exactly(*[matches_tensor(e) for e in expected])


def assert_tensor_sequence_equals(
    actual: Sequence[torch.Tensor],
    *expected: TensorConvertable,
    view_of: Optional[torch.Tensor] = None,
) -> Sequence[torch.Tensor]:
    """Assert that the `actual` is a sequence that equals the given `expected`
    tensor values.

    :param actual: the `actual` to test.
    :param expected: the expected values.
    :param view_of: if present, also check that actual is a view of the reference Tensor.
    :return: the `actual`
    """
    hamcrest.assert_that(
        actual,
        match_tensor_sequence(*expected),
    )
    if view_of is not None:
        assert_tensor_views(view_of, *actual)
    return actual


@contextlib.contextmanager
def reset_generator_seed(seed: int = 3 * 17 * 53 + 1) -> Iterator[None]:
    """Context manager which resets the `torch.manual_seed()` seed on entry.

    :param seed: optional seed.
    """
    torch.manual_seed(seed)
    yield
