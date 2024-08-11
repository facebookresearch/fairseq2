# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields
from enum import Enum
from hashlib import sha1
from typing import Any, Final

from fairseq2.typing import DataClass, DataType, is_dataclass_instance


class SweepTagger:
    """Generates a sweep tag from the diff of two recipe configurations."""

    _DEFAULT_ALLOW_SET: Final = {
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

    def __init__(self, *, allow_set: set[str] | None = None) -> None:
        """
        :param allow_set:
            The configuration field names allowed while generating the sweep tag.
        """
        if allow_set is None:
            allow_set = self._DEFAULT_ALLOW_SET.copy()

        self._allow_set = allow_set

    def extend_allow_set(self, *extras: str) -> None:
        """Extend the allowed configuration field names with ``extras``."""
        self._allow_set.update(extras)

    def __call__(self, preset: str, preset_config: DataClass, config: DataClass) -> str:
        """
        :param preset:
            The name of the preset recipe.
        :param preset_config:
            The preset (i.e. ground-truth) recipe configuration.
        :param config:
            The recipe configuration for which to generate a sweep tag.
        """
        if type(config) is not type(preset_config):
            raise ValueError(
                f"`config` must be of the same type as `preset_config` (`{type(preset_config)}`), but is of type `{type(config)}` instead."
            )

        output = [f"preset_{self._remove_non_word(preset)}"]

        try:
            world_size = os.environ["WORLD_SIZE"]
        except KeyError:
            world_size = "1"

        output.append(f"ws_{world_size}")

        def abbrv(s: str) -> str:
            if s.startswith("num_"):
                s = f"n_{s[4:]}"

            return s

        def generate(config: DataClass) -> None:
            for field in fields(config):
                value = getattr(config, field.name)

                if is_dataclass_instance(value):
                    generate(config)
                elif field.name in self._allow_set:
                    if s := self._to_tag_value(value):
                        output.append(f"{abbrv(field.name)}_{s}")

        def generate_from_diff(preset_config: DataClass, config: DataClass) -> None:
            for field in fields(config):
                value = getattr(config, field.name)

                preset_value = getattr(preset_config, field.name)

                if is_dataclass_instance(preset_value):
                    if type(value) is type(preset_value):
                        generate_from_diff(preset_value, value)
                    else:
                        generate(value)
                elif field.name in self._allow_set:
                    if preset_value == value:
                        continue

                    if s := self._to_tag_value(value):
                        output.append(f"{abbrv(field.name)}_{s}")

        generate_from_diff(preset_config, config)

        s = ".".join(output)

        # Cap to maximum of 128 characters.
        if len(s) > 128:
            # Make sure we avoid name conflicts by prepending the hash of the
            # whole tag to the truncated one.
            s = s[:120] + self._hash(s)

        return s

    @classmethod
    def _to_tag_value(cls, value: Any) -> str | None:
        s: str | None

        if isinstance(value, str):
            s = cls._remove_non_word(value)

            if len(s) < 16:
                return s

            return cls._hash(s)

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
                if s := cls._to_tag_value(v):
                    output.append(s)

            if not output:
                return None

            s = "-".join(output)

            return f"b{s}e"

        if isinstance(value, Mapping):
            output = []

            for k, v in value.items():
                ks = cls._to_tag_value(k)
                vs = cls._to_tag_value(v)

                if ks and vs:
                    output.append(f"{ks}_{vs}")

            if not output:
                return None

            output.sort()

            s = "-".join(output)

            return f"b{s}e"

        return None

    @staticmethod
    def _remove_non_word(s: str) -> str:
        return re.sub(r"[^-_\w]", "", s)

    @staticmethod
    def _hash(s: str) -> str:
        s = sha1(s.encode("utf-8")).hexdigest()

        return s[:8]


default_sweep_tagger = SweepTagger()
