# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable, Sequence, Set
from enum import Enum
from functools import cache
from hashlib import sha1
from typing import final

from fairseq2.error import InternalError
from fairseq2.typing import EMPTY
from fairseq2.utils.structured import StructureError


@final
class SweepTagGenerator:
    _world_size: int
    _allowed_keys: Set[Hashable]
    _format: str

    def __init__(
        self, world_size: int, allowed_keys: Set[Hashable], fmt: str | None = None
    ) -> None:
        """
        :param allowed_keys: The recipe configuration keys allowed to be used in
            sweep tags.
        """
        self._world_size = world_size
        self._allowed_keys = allowed_keys

        if fmt is None:
            self._format = "ps_{preset}.ws_{world_size}.{hash}"
        else:
            self._format = fmt.strip()
            if not self._format:
                raise ValueError("`fmt` must not be empty.")

            self._safe_format({}, dry_run=True)

    def generate(self, preset: str, unstructured_config: object) -> str:
        tags = {"preset": preset, "world_size": f"{self._world_size}"}

        self._collect_tags(unstructured_config, tags, path="")

        tags["hash"] = self._generate_hash(tags)

        return self._safe_format(tags, dry_run=False)

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

    def _safe_format(self, tags: dict[str, str], dry_run: bool) -> str:
        class State(Enum):
            LITERAL = 0
            PLACEHOLDER = 1
            OPENING_BRACE = 2
            CLOSING_BRACE = 3

        output = []

        placeholder: list[str] = []

        unknown_keys: set[str] = set()

        state = State.LITERAL

        for c in self._format:
            match state:
                case State.LITERAL:
                    if c == "{":
                        state = State.OPENING_BRACE
                    elif c == "}":
                        state = State.CLOSING_BRACE
                    elif not dry_run:
                        output.append(c)
                case State.OPENING_BRACE:
                    if c == "{":  # escape
                        state = State.LITERAL

                        if not dry_run:
                            output.append("{")
                    elif c == "}":
                        raise SweepFormatError(
                            "`fmt` must not have any empty placeholders."
                        )
                    else:
                        state = State.PLACEHOLDER

                        if not dry_run:
                            placeholder.append(c)
                case State.PLACEHOLDER:
                    if c == "}":
                        state = State.LITERAL

                        if not dry_run:
                            key = "".join(placeholder)

                            tag: Iterable[str] | None = tags.get(key)
                            if tag is None:
                                tag = placeholder

                                unknown_keys.add(key)

                            output.extend(tag)

                            placeholder.clear()
                    elif not dry_run:
                        placeholder.append(c)
                case State.CLOSING_BRACE:
                    state = State.LITERAL

                    if not dry_run:
                        if c == "}":  # escape
                            output.append("}")
                        else:
                            output.append(c)

        if state != State.LITERAL:
            raise SweepFormatError(
                "`fmt` must have matching opening and closing braces."
            )

        if unknown_keys:
            if dry_run:
                raise InternalError(
                    "`dry_run` is set, but `unknown_keys` is not empty."
                )

            keys = list(unknown_keys)

            keys.sort()

            s = ", ".join(keys)

            raise SweepFormatPlaceholderError(
                keys, f"The sweep format string must contain only placeholders that correspond to the configuration keys, but contains the following unexpected placeholder(s): {s}"  # fmt: skip
            )

        if dry_run:
            return ""

        return "".join(output)


class SweepFormatError(ValueError):
    pass


class SweepFormatPlaceholderError(ValueError):
    unknown_keys: Sequence[str]

    def __init__(self, unknown_keys: Sequence[str], message: str) -> None:
        super().__init__(message)

        self.unknown_keys = unknown_keys


def get_sweep_keys(extra_sweep_keys: Set[Hashable] | None) -> Set[Hashable]:
    sweep_keys = get_default_sweep_keys()

    if extra_sweep_keys is not None:
        sweep_keys = sweep_keys | extra_sweep_keys

    return sweep_keys


@cache
def get_default_sweep_keys() -> Set[Hashable]:
    return {
        "batch_shuffle_window",
        "betas",
        "data_parallelism",
        "dataset",
        "dtype",
        "example_shuffle_window",
        "final_lr_ratio",
        "final_lr_scale",
        "fp16_loss_scale",
        "fsdp_reshard_after_forward",
        "fsdp_wrap_granularity",
        "gradient_accumulation",
        "label_smoothing",
        "lr",
        "lr_stage_ratios",
        "max_gradient_norm",
        "max_num_elements",
        "max_num_steps",
        "max_num_tokens",
        "max_seq_len",
        "mixed_precision",
        "model",
        "model_arch",
        "model_config",
        "num_lr_warmup_steps",
        "pretrained_model",
        "seed",
        "split",
        "start_lr",
        "start_lr_scale",
        "tensor_parallel_size",
        "tokenizer",
        "train_split",
        "valid_split",
        "weight_decay",
    }
