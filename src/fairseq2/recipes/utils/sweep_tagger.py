# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Sequence, Set
from enum import Enum
from hashlib import sha1
from typing import final

from typing_extensions import override

from fairseq2.utils.dataclass import EMPTY
from fairseq2.utils.structured import StructureError


class SweepTagger(ABC):
    """Generates a sweep tag from a recipe configuration."""

    @abstractmethod
    def generate(
        self,
        world_size: int,
        preset: str,
        unstructured_config: object,
        fmt: str | None = None,
    ) -> str:
        ...


@final
class StandardSweepTagger(SweepTagger):
    _allowed_keys: Set[Hashable]

    def __init__(self, allowed_keys: Set[Hashable]) -> None:
        """
        :param allowed_keys: The recipe configuration keys allowed to be used in
            sweep tags.
        """
        self._allowed_keys = allowed_keys

    @override
    def generate(
        self,
        world_size: int,
        preset: str,
        unstructured_config: object,
        fmt: str | None = None,
    ) -> str:
        if fmt is None:
            fmt = "ps_{preset}.ws_{world_size}.{hash}"
        else:
            fmt = fmt.strip()
            if not fmt:
                raise SweepFormatError("`fmt` must not be empty.")

        tags = {"preset": preset, "world_size": f"{world_size}"}

        self._collect_tags(unstructured_config, tags, path="")

        tags["hash"] = self._generate_hash(tags)

        return self._safe_format(fmt, tags)

    def _collect_tags(self, obj: object, tags: dict[str, str], path: str) -> None:
        if obj is None:
            tags[path] = "none"

            return

        if obj is EMPTY:
            tags[path] = "empty"

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
                if key in self._allowed_keys:
                    self._collect_tags(
                        value, tags, path=f"{path}.{key}" if path else f"{key}"
                    )

            return

        raise StructureError(
            "`unstructured_config` must be of a composition of types `bool`, `int`, `float`, `str`, `list`, and `dict`."
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

    @staticmethod
    def _safe_format(fmt: str, tags: dict[str, str]) -> str:
        class State(Enum):
            LITERAL = 0
            PLACEHOLDER = 1
            OPENING_BRACE = 2
            CLOSING_BRACE = 3

        output = []

        placeholder: list[str] = []

        unknown_keys: set[str] = set()

        state = State.LITERAL

        for c in fmt:
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
                        raise SweepFormatError(
                            "`fmt` must not have any empty placeholders"
                        )
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
            raise SweepFormatError(
                "`fmt` must have matching opening and closing braces."
            )

        if unknown_keys:
            keys = list(unknown_keys)

            keys.sort()

            s = ", ".join(keys)

            raise SweepFormatPlaceholderError(
                keys, f"`fmt` must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder(s): {s}"  # fmt: skip
            )

        return "".join(output)


@final
class NoopSweepTagger(SweepTagger):
    @override
    def generate(
        self,
        world_size: int,
        preset: str,
        unstructured_config: object,
        fmt: str | None = None,
    ) -> str:
        return ""


class SweepFormatError(ValueError):
    pass


class SweepFormatPlaceholderError(SweepFormatError):
    unknown_keys: Sequence[str]

    def __init__(self, unknown_keys: Sequence[str], message: str) -> None:
        super().__init__(message)

        self.unknown_keys = unknown_keys
