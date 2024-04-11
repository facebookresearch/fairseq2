# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, List, MutableMapping

import yaml
from yaml.representer import RepresenterError


def update_dataclass(obj: Any, overrides: MutableMapping[str, Any]) -> None:
    """Update ``obj`` with the data contained in ``overrides``.

    :param obj:
        The object to update. Must be a :class:`~dataclasses.dataclass`.
    :param overrides:
        The dictionary containing the data to set in ``obj``.
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
            f"`obj` must contain the following keys that are present in `overrides`: {leftovers}"
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


def _dump_dataclass(obj: Any, file: Path) -> None:
    if not is_dataclass(obj):
        raise ValueError(
            f"`obj` must be a ` dataclass`, but is of type `{type(obj)}` instead."
        )

    fp = file.open("w")

    try:
        yaml.safe_dump(asdict(obj), fp)
    except RepresenterError as ex:
        raise ValueError(
            "`obj` must contain values of only primitive stdlib types, other dataclasses, and types registered with `yaml.representer.SafeRepresenter`. See nested exception for details."
        ) from ex
    finally:
        fp.close()
