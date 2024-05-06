# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, cast

import yaml

from fairseq2.typing import DataClass, DataType, Device, is_dataclass_instance


def update_dataclass(obj: DataClass, overrides: MutableMapping[str, Any]) -> None:
    """Update ``obj`` with the data contained in ``overrides``.

    :param obj:
        The data class instance to update.
    :param overrides:
        The dictionary containing the data to set in ``obj``.
    """
    leftovers: List[str] = []

    _do_update_dataclass(obj, overrides, leftovers, path=[])

    if leftovers:
        leftovers.sort()

        raise ValueError(
            f"`overrides` must contain only keys that are present in `obj`, but the following keys do not exist in `obj`: {leftovers}"
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


def dump_dataclass(obj: DataClass, file: Path) -> None:
    """Dump ``obj`` to ``file`` in YAML format."""
    safe_data = to_safe_dict(obj)

    with file.open("w") as fp:
        yaml.safe_dump_all(safe_data, fp)


def to_safe_dict(obj: DataClass) -> Dict[str, Any]:
    """Convert ``obj`` to a :class:`dict` safe to serialize in YAML/JSON."""

    def _convert(d: Any) -> Any:
        if d is None:
            return d

        if isinstance(d, (bool, int, float, str)):
            return d

        if isinstance(d, (list, tuple)):
            return [_convert(e) for e in d]

        if isinstance(d, dict):
            return {_convert(k): _convert(v) for k, v in d.items()}

        if isinstance(d, Enum):
            return d.name

        if isinstance(d, Path):
            return str(d)

        if isinstance(d, DataType):
            return str(d)[6:]  # Strip 'torch.'.

        if isinstance(d, Device):
            return str(d)

        raise ValueError(
            "`obj` must contain values of only primitive types, lists, and dictionaries."
        )

    return cast(Dict[str, Any], _convert(asdict(obj)))
