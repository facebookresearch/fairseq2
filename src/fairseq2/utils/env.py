# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import cast

from fairseq2.runtime.dependency import DependencyResolver


def get_env(resolver: DependencyResolver) -> MutableMapping[str, str]:
    env = resolver.resolve(MutableMapping, key="env")

    return cast(MutableMapping[str, str], env)


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
    except ValueError as ex:
        raise InvalidEnvironmentVariableError(
            name, f"The value of the `{name}` environment variable cannot be parsed as an integer. See the nested exception for details."  # fmt: skip
        ) from ex

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
    except ValueError as ex:
        raise InvalidEnvironmentVariableError(
            name, f"The value of the `{name}` environment variable cannot be parsed as a pathname. See the nested exception for details."  # fmt: skip
        ) from ex


class InvalidEnvironmentVariableError(Exception):
    name: str

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)

        self.name = name


def get_rank(env: Mapping[str, str]) -> int:
    value = get_int_from_env(env, "RANK", allow_zero=True)

    return 0 if value is None else value


def get_world_size(env: Mapping[str, str]) -> int:
    value = get_int_from_env(env, "WORLD_SIZE")

    return 1 if value is None else value


def get_local_rank(env: Mapping[str, str]) -> int:
    value = get_int_from_env(env, "LOCAL_RANK", allow_zero=True)

    return 0 if value is None else value


def get_local_world_size(env: Mapping[str, str]) -> int:
    value = get_int_from_env(env, "LOCAL_WORLD_SIZE")

    return 1 if value is None else value
