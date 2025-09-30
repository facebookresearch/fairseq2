# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from hashlib import sha1
from typing import final

from typing_extensions import override

from fairseq2.error import InternalError
from fairseq2.recipe.config import CommonSection, RecipeConfig
from fairseq2.utils.structured import StructureError, ValueConverter
from fairseq2.utils.validation import ValidationError
from fairseq2.world_info import WorldInfo


class _SweepTagGenerator(ABC):
    @abstractmethod
    def maybe_generate(self) -> str | None: ...


@final
class _StandardSweepTagGenerator(_SweepTagGenerator):
    def __init__(
        self,
        section: CommonSection,
        world_info: WorldInfo,
        config: RecipeConfig,
        value_converter: ValueConverter,
    ) -> None:
        self._section = section
        self._world_info = world_info
        self._config = config
        self._value_converter = value_converter

    @override
    def maybe_generate(self) -> str | None:
        if self._section.no_sweep_dir:
            return None

        untyped_config = self._config.as_(object)

        try:
            unstructured_config = self._value_converter.unstructure(untyped_config)
        except StructureError as ex:
            raise InternalError(
                "`self._config` cannot be converted to an unstructured form."
            ) from ex

        sweep_format = self._section.sweep_format.strip()
        if not sweep_format:
            msg = "`sweep_format` must not be empty."

            raise ValidationError(msg, field="common")

        tags = {"world_size": f"{self._world_info.size}"}

        self._collect_tags(unstructured_config, tags, path="")

        tags["hash"] = self._generate_hash(tags)

        return self._safe_format(sweep_format, tags)

    def _collect_tags(self, obj: object, tags: dict[str, str], path: str) -> None:
        if obj is None:
            tags[path] = "none"

            return

        if isinstance(obj, str):
            tag = self._remove_non_word(obj)

            if len(tag) >= 16:
                tag = self._generate_tag_hash(tag)

            tags[path] = tag

            return

        if isinstance(obj, bool):
            tags[path] = "t" if obj else "f"

            return

        if isinstance(obj, int | float):
            tags[path] = f"{obj}"

            return

        if isinstance(obj, list):
            for idx, elem in enumerate(obj):
                self._collect_tags(elem, tags, path=f"{path}[{idx}]")

            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                self._collect_tags(
                    value, tags, path=f"{path}.{key}" if path else f"{key}"
                )

            return

        raise InternalError(
            f"Unstructured `config` contains an object of type `{type(obj)}`."
        )

    @staticmethod
    def _remove_non_word(s: str) -> str:
        return re.sub(r"[^-_\w]", "", s)

    @staticmethod
    def _generate_tag_hash(s: str) -> str:
        algo = sha1(s.encode("utf-8"))

        h = algo.hexdigest()

        return h[:8]

    @staticmethod
    def _generate_hash(tags: dict[str, str]) -> str:
        algo = sha1()

        for k, v in sorted(tags.items()):
            algo.update(k.encode("utf-8"))
            algo.update(v.encode("utf-8"))

        h = algo.hexdigest()

        return h[:8]

    def _safe_format(self, sweep_format: str, tags: dict[str, str]) -> str:
        class State(Enum):
            LITERAL = 0
            PLACEHOLDER = 1
            OPENING_BRACE = 2
            CLOSING_BRACE = 3

        output = []

        placeholder: list[str] = []

        unknown_keys: set[str] = set()

        state = State.LITERAL

        for c in sweep_format:
            match state:
                case State.LITERAL:
                    if c == "{":
                        state = State.OPENING_BRACE
                    elif c == "}":
                        state = State.CLOSING_BRACE
                    else:
                        output.append(c)
                case State.OPENING_BRACE:
                    if c == "{":  # escape
                        state = State.LITERAL

                        output.append("{")
                    elif c == "}":
                        msg = "`sweep_format` must not have any empty placeholders."

                        raise ValidationError(msg, field="common")
                    else:
                        state = State.PLACEHOLDER

                        placeholder.append(c)
                case State.PLACEHOLDER:
                    if c == "}":
                        state = State.LITERAL

                        key = "".join(placeholder)

                        tag: Iterable[str] | None = tags.get(key)
                        if tag is None:
                            tag = placeholder

                            unknown_keys.add(key)

                        output.extend(tag)

                        placeholder.clear()
                    else:
                        placeholder.append(c)
                case State.CLOSING_BRACE:
                    state = State.LITERAL

                    if c == "}":  # escape
                        output.append("}")
                    else:
                        output.append(c)

        if state != State.LITERAL:
            msg = "`sweep_format` must have matching opening and closing braces."

            raise ValidationError(msg, field="common")

        if unknown_keys:
            keys = list(unknown_keys)

            keys.sort()

            s = ", ".join(keys)

            msg = f"`sweep_format` must contain only placeholders that correspond to the configuration keys, but contains unexpected placeholder(s) {s}."

            raise ValidationError(msg, field="common")

        return "".join(output)
