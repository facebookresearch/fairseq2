# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, MutableMapping

import yaml
from typing_extensions import TypeGuard
from yaml.representer import RepresenterError

from fairseq2.typing import DataClass


def is_dataclass_instance(obj: Any) -> TypeGuard[DataClass]:
    """Return ``True`` if ``obj`` is of type :class:`DataClass`."""
    return is_dataclass(obj) and not isinstance(obj, type)


def update_dataclass(obj: DataClass, overrides: MutableMapping[str, Any]) -> None:
    """Update ``obj`` with the data contained in ``overrides``.

    :param obj:
        The object to update.
    :param overrides:
        The dictionary containing the data to set in ``obj``.
    """
    leftovers: List[str] = []

    _do_update_dataclass(obj, overrides, leftovers, path=[])

    if leftovers:
        leftovers.sort()

        raise ValueError(
            f"`overrides` contains the following keys that are not present in `obj`: {leftovers}"
        )


def _do_update_dataclass(
    obj: DataClass,
    overrides: MutableMapping[str, Any],
    leftovers: List[str],
    path: List[str],
) -> None:
    for name, value in obj.__dict__.items():
        try:
            override = overrides.pop(name)
        except KeyError:
            continue

        # Recursively traverse child dataclasses.
        if override is not None and is_dataclass_instance(value):
            if not isinstance(override, MutableMapping):
                p = ".".join(path + [name])

                raise TypeError(
                    f"The value of the key '{p}' in `overrides` must be of a mapping type (e.g. `dict`), but is of type `{type(override)}` instead."
                )

            path.append(name)

            _do_update_dataclass(value, override, leftovers, path)

            path.pop()
        else:
            if value is not None and not isinstance(override, kls := type(value)):
                override = _maybe_convert(override, kls)

            setattr(obj, name, override)

    if overrides:
        leftovers += [".".join(path + [name]) for name in overrides]


def _maybe_convert(value: Any, kls: type) -> Any:
    if issubclass(kls, Enum) and isinstance(value, str):
        try:
            return kls[value]
        except KeyError:
            pass

    return value


def _dump_dataclass(obj: DataClass, file: Path) -> None:
    fp = file.open("w")

    try:
        yaml.safe_dump(asdict(obj), fp)
    except RepresenterError as ex:
        raise ValueError(
            "`obj` must contain values of only primitive stdlib types and types registered with `yaml.representer.SafeRepresenter`. See nested exception for details."
        ) from ex
    finally:
        fp.close()
