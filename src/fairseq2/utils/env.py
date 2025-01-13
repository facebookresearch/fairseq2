# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from pathlib import Path

from fairseq2.logging import log
from fairseq2.typing import Device


def get_int_from_env(var_name: str, allow_zero: bool = False) -> int | None:
    """Return the value of an environment variable as ``int``.

    :param var_name:
        The name of the environment variable.
    :param allow_zero:
        If ``True``, returns the value if it equals to zero; otherwise, raises
        a :class:`InvalidEnvironmentVariableError`.
    """
    s = os.environ.get(var_name)
    if s is None:
        return None

    try:
        value = int(s)
    except ValueError:
        raise InvalidEnvironmentVariableError(
            f"The value of the `{var_name}` environment variable is expected to be an integer, but is '{s}' instead."
        ) from None

    if not allow_zero:
        if not value >= 1:
            raise InvalidEnvironmentVariableError(
                f"The value of the `{var_name}` environment variable is expected to be a positive integer, but is {value} instead."
            )
    else:
        if not value >= 0:
            raise InvalidEnvironmentVariableError(
                f"The value of the `{var_name}` environment variable is expected to be greater than or equal to 0, but is {value} instead."
            )

    return value


def get_path_from_env(var_name: str, missing_ok: bool = False) -> Path | None:
    """Return the value of an environment variable as :class:`~pathlib.Path`.

    :param var_name:
        The name of the environment variable.
    :param log:
        The log to write to.
    :param missing_ok:
        If ``True``, returns ``None`` if the path does not exist; otherwise,
        raises a :class:`InvalidEnvironmentVariableError`.
    """
    pathname = os.environ.get(var_name)
    if not pathname:
        return None

    try:
        path = Path(pathname)
    except ValueError:
        raise InvalidEnvironmentVariableError(
            f"The value of the `{var_name}` environment variable is expected to be a pathname, but is '{pathname}' instead."
        ) from None

    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        if missing_ok:
            return resolved_path

        log.warning("The '{}' path pointed to by the `{}` environment variable does not exist.", path, var_name)  # fmt: skip

        return None

    return resolved_path


def get_device_from_env(var_name: str) -> Device | None:
    device_str = os.environ.get(var_name)
    if device_str is None:
        return None

    try:
        return Device(device_str)
    except (RuntimeError, ValueError):
        raise InvalidEnvironmentVariableError(
            f"The value of the `{var_name}` environment variable is expected to specify a PyTorch device, but is '{device_str}' instead."
        ) from None


class InvalidEnvironmentVariableError(Exception):
    pass
