# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from enum import Enum
from hashlib import sha1
from typing import final

from typing_extensions import override

from fairseq2.utils.dataclass import EMPTY
from fairseq2.utils.structured import StructureError


class SweepTagger(ABC):
    """Generates a sweep tag from a recipe configuration."""

    @abstractmethod
    def __call__(self, preset: str, unstructured_config: object) -> str:
        """
        :param preset:
            The name of the preset recipe.
        :param unstructured_config:
            The unstructured configuration of the preset recipe.
        """

    @abstractmethod
    def extend_allow_set(self, *keys: Hashable) -> None:
        """Extend the allowed configuration keys with ``keys``."""


class SweepFormatError(ValueError):
    pass


@final
class StandardSweepTagger(SweepTagger):
    _allow_set: set[object]

    def __init__(self, *, allow_set: set[object] | None = None) -> None:
        """
        :param allow_set:
            The configuration keys allowed to be used in sweep tags.
        """
        if allow_set is not None:
            self._allow_set = allow_set
        else:
            self._allow_set = {
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

    @override
    def __call__(self, preset: str, unstructured_config: object) -> str:
        try:
            world_size = os.environ["WORLD_SIZE"]
        except KeyError:
            world_size = "1"

        tags = {"preset": preset, "world_size": world_size}

        self._collect_tags(unstructured_config, tags, path="")

        sweep_format = tags.pop("sweep_format", None)
        if sweep_format is None:
            sweep_format = "ps_{preset}.ws_{world_size}.{hash}"

        tags["hash"] = self._generate_hash(tags)

        return self._safe_format(sweep_format, tags)

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
                if path == "" and key == "sweep_format":
                    if not isinstance(value, str):
                        raise SweepFormatError(
                            "The 'sweep_format' key of `unstructured_config` must be of type `str`."
                        )

                    tags[key] = value
                elif key in self._allow_set:
                    self._collect_tags(
                        value, tags, path=f"{path}.{key}" if path else f"{key}"
                    )

            return

        raise StructureError(
            "`unstructured_config` must be a composition of types `bool`, `int`, `float`, `str`, `list`, and `dict`."
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
    def _safe_format(sweep_format: str, tags: dict[str, str]) -> str:
        class State(Enum):
            LITERAL = 0
            PLACEHOLDER = 1
            OPENING_BRACE = 2
            CLOSING_BRACE = 3

        output = []

        placeholder: list[str] = []

        missing_keys: set[str] = set()

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
                    else:
                        state = State.PLACEHOLDER

                        placeholder.append(c)
                case State.PLACEHOLDER:
                    if c == "}":
                        state = State.LITERAL

                        key = "".join(placeholder)

                        tag: Iterable[str]

                        try:
                            tag = tags[key]
                        except KeyError:
                            tag = placeholder

                            missing_keys.add(key)

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

        if state == State.OPENING_BRACE or state == State.PLACEHOLDER:
            raise SweepFormatError(
                "The 'sweep_format' key of `unstructured_config` is not a valid format string."
            )

        if missing_keys:
            missing_key_list = list(missing_keys)

            missing_key_list.sort()

            s = ", ".join(missing_key_list)

            raise SweepFormatError(
                f"The 'sweep_format' key of `unstructured_config` contains the following placeholders that do not correspond to any key in the configuration: {s}"
            )

        return "".join(output)

    @override
    def extend_allow_set(self, *keys: Hashable) -> None:
        self._allow_set.update(keys)


default_sweep_tagger: SweepTagger = StandardSweepTagger()
