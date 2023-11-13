# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import is_dataclass
from typing import Any, List, MutableMapping


def update_dataclass(obj: Any, overrides: MutableMapping[str, Any]) -> None:
    """Update the specified data class with the data contained in ``overrides``.

    :param obj:
        A :type:`dataclasses.dataclass` object.
    :param overrides:
        A dictionary containing the data to set in ``obj``.
    """
    if not is_dataclass(obj):
        raise TypeError(
            f"`obj` must be a `dataclass`, but is of type `{type(obj)}` instead."
        )

    leftovers: List[str] = []

    _do_update_dataclass(obj, overrides, leftovers, path=[])

    if leftovers:
        leftovers.sort()

        raise ValueError(
            f"The following keys contained in `overrides` do not exist in `obj`: {leftovers}"
        )


def _do_update_dataclass(
    obj: Any, overrides: MutableMapping[str, Any], leftovers: List[str], path: List[str]
) -> None:
    for name, value in obj.__dict__.items():
        try:
            override = overrides.pop(name)
        except KeyError:
            continue

        # Recursively traverse child dataclasses.
        if override is not None and is_dataclass(value):
            if not isinstance(override, MutableMapping):
                p = ".".join(path + [name])

                raise TypeError(
                    f"The key '{p}' must be of a mapping type (e.g. `dict`), but is of type `{type(override)}` instead."
                )

            path.append(name)

            _do_update_dataclass(value, override, leftovers, path)

            path.pop()
        else:
            setattr(obj, name, override)

    if overrides:
        leftovers += [".".join(path + [name]) for name in overrides]
