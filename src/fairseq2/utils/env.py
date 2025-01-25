# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fairseq2.typing import Device


def get_int_from_env(
    env: Mapping[str, str], name: str, allow_zero: bool = False
) -> int | None:
    """Return the value of an environment variable as ``int``.

    :param name: The name of the environment variable.
    :param allow_zero: If ``True``, returns the value if it equals to zero;
        otherwise, raises a :class:`InvalidEnvironmentVariableError`.
    """
    s = env.get(name)
    if s is None:
        return None

    try:
        value = int(s)
    except ValueError:
        raise InvalidEnvironmentVariableError(
            name, f"The value of the `{name}` environment variable is expected to be an integer, but is '{s}' instead."  # fmt: skip
        ) from None

    if not allow_zero:
        if not value >= 1:
            raise InvalidEnvironmentVariableError(
                name, f"The value of the `{name}` environment variable is expected to be a positive integer, but is {value} instead."  # fmt: skip
            )
    else:
        if not value >= 0:
            raise InvalidEnvironmentVariableError(
                name, f"The value of the `{name}` environment variable is expected to be greater than or equal to 0, but is {value} instead."  # fmt: skip
            )

    return value


def get_path_from_env(env: Mapping[str, str], name: str) -> Path | None:
    """Return the value of an environment variable as :class:`~pathlib.Path`.

    :param name: The name of the environment variable.
    """
    pathname = env.get(name)
    if not pathname:
        return None

    try:
        return Path(pathname)
    except ValueError:
        raise InvalidEnvironmentVariableError(
            name, f"The value of the `{name}` environment variable is expected to be a pathname, but is '{pathname}' instead."  # fmt: skip
        ) from None


def get_device_from_env(env: Mapping[str, str], name: str) -> Device | None:
    device_str = env.get(name)
    if device_str is None:
        return None

    try:
        return Device(device_str)
    except (RuntimeError, ValueError):
        raise InvalidEnvironmentVariableError(
            name, f"The value of the `{name}` environment variable is expected to specify a PyTorch device, but is '{device_str}' instead."  # fmt: skip
        ) from None


def get_world_size(env: Mapping[str, str]) -> int:
    value = get_int_from_env(env, "WORLD_SIZE")

    return 1 if value is None else value


def get_rank(env: Mapping[str, str]) -> int:
    value = get_int_from_env(env, "RANK", allow_zero=True)

    return 0 if value is None else value


def get_local_world_size(env: Mapping[str, str]) -> int:
    value = get_int_from_env(env, "LOCAL_WORLD_SIZE")

    return 1 if value is None else value


def get_local_rank(env: Mapping[str, str]) -> int:
    value = get_int_from_env(env, "LOCAL_RANK", allow_zero=True)

    return 0 if value is None else value


class InvalidEnvironmentVariableError(Exception):
    name: str

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name
