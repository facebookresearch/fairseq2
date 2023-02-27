# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast


def get_name(fn: Any) -> str:
    """Return the printable, user-friendly name of ``fn``.

    :param fn:
        The callable to inspect.
    """
    if not callable(fn):
        raise ValueError("`fn` must be a callable.")

    name = getattr(fn, "__name__", None)
    if name is not None:
        return cast(str, name)

    # Check if the function is wrapped by functools.partial and, if so, try to
    # get the wrapped name.
    wrapped_fn = getattr(fn, "func", None)
    if wrapped_fn is not None:
        return get_name(wrapped_fn)

    # Otherwise, fall back to the type name.
    return type(fn).__name__
