# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import fields
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from fairseq2.typing import DataClass, DataType, is_dataclass_instance


def generate_sweep_tag(preset: str, preset_config: DataClass, config: DataClass) -> str:
    """Generate a sweep tag from the diff of ``preset_config`` and ``config``."""
    if type(config) is not type(preset_config):
        raise ValueError(
            f"`config` must be of the same type as `preset_config` (`{type(preset_config)}`), but is of type `{type(config)}` instead."
        )

    output = [_remove_non_word(preset)]

    def generate(config: DataClass) -> None:
        for field in fields(config):
            value = getattr(config, field.name)

            if is_dataclass_instance(value):
                generate(config)
            else:
                if s := _to_tag_value(value):
                    output.append(f"{field.name}_{s}")

    def generate_from_diff(preset_config: DataClass, config: DataClass) -> None:
        for field in fields(config):
            value = getattr(config, field.name)

            preset_value = getattr(preset_config, field.name)

            if is_dataclass_instance(preset_value):
                if type(value) is type(preset_value):
                    generate_from_diff(preset_value, value)
                else:
                    generate(value)
            else:
                if preset_value == value:
                    continue

                if s := _to_tag_value(value):
                    output.append(f"{field.name}_{s}")

    generate_from_diff(preset_config, config)

    s = ".".join(output)

    return s[:256]  # Cap to maximum 256 characters.


def _to_tag_value(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return _remove_non_word(value)

    if isinstance(value, bool):
        return "t" if value else "f"

    if isinstance(value, (int, float)):
        return f"{value}"

    if isinstance(value, DataType):
        return f"{value}"[6:]

    if isinstance(value, Enum):
        return value.name

    if isinstance(value, Sequence):
        output = []

        for v in value:
            if s := _to_tag_value(v):
                output.append(s)

        if not output:
            return None

        s = "-".join(output)

        return f"b{s}e"

    if isinstance(value, Mapping):
        output = []

        for k, v in value.items():
            ks = _to_tag_value(k)
            vs = _to_tag_value(v)

            if ks and vs:
                output.append(f"{ks}_{vs}")

        if not output:
            return None

        output.sort()

        s = "-".join(output)

        return f"b{s}e"

    return None


def _remove_non_word(s: str) -> str:
    return re.sub(r"[^\w]", "", s)
