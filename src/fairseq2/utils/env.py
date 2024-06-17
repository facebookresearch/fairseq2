# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Optional

from fairseq2.logging import LogWriter


def get_int_from_env(var_name: str, allow_zero: bool = False) -> Optional[int]:
    """Return the value of an environment variable as ``int``.

    :param var_name:
        The name of the environment variable.
    :param allow_zero:
        If ``True``, returns the value if it equals to zero; otherwise, raises
        a :class:`RuntimeError`.
    """
    s = os.environ.get(var_name)
    if s is None:
        return None

    try:
        value = int(s)
    except ValueError:
        raise RuntimeError(
            f"The value of the `{var_name}` environment variable must be an integer, but is '{s}' instead."
        )

    if not allow_zero:
        if not value >= 1:
            raise RuntimeError(
                f"The value of the `{var_name}` environment variable must be greater than 0, but is {value} instead."
            )
    else:
        if not value >= 0:
            raise RuntimeError(
                f"The value of the `{var_name}` environment variable must be greater than or equal to 0, but is {value} instead."
            )

    return value


def get_path_from_env(
    var_name: str, log: LogWriter, missing_ok: bool = False
) -> Optional[Path]:
    """Return the value of an environment variable as :class:`~pathlib.Path`.

    :param var_name:
        The name of the environment variable.
    :param log:
        The log to write to.
    :param missing_ok:
        If ``True``, returns ``None`` if the path does not exist; otherwise,
        raises a :class:`RuntimeError`.
    """
    pathname = os.environ.get(var_name)
    if not pathname:
        return None

    try:
        path = Path(pathname)
    except ValueError as ex:
        raise RuntimeError(
            f"The value of the `{var_name}` environment variable must be a pathname, but is '{pathname}' instead."
        ) from ex

    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        if missing_ok:
            return resolved_path

        log.warning("The path '{}' pointed to by the `{}` environment variable does not exist.", path, var_name)  # fmt: skip

        return None

    return resolved_path
