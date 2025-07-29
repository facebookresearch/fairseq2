# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Dict

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.utils.progress import ProgressReporter


def load_checkpoint(
    model: Module,
    checkpoint: Iterable[tuple[str, Tensor]],
    progress_reporter: ProgressReporter,
) -> None:
    errors = []

    unexpected_keys = []

    memo = set()

    # To determine shared tensors we have to preserve the identity of the
    # parameters and buffers via `keep_vars`.
    state_dict = model.state_dict(keep_vars=True)

    with progress_reporter:
        # Start the progress bar in indeterminate mode since the first iteration
        # typically takes longer to handle; in particular when the checkpoint is
        # not mmap'ed.
        progress_task = progress_reporter.create_task(
            name="parameter load", total=len(state_dict), start=False
        )

        with progress_task, torch.no_grad():
            for key, tensor in checkpoint:
                try:
                    state_tensor = state_dict.pop(key)
                except KeyError:
                    unexpected_keys.append(key)

                    continue

                progress_task.start()

                if tensor.shape != state_tensor.shape:
                    errors.append(
                        f"`{key}` has a shape of {tuple(tensor.shape)} in the checkpoint, but has a shape of {tuple(state_tensor.shape)} in the model."
                    )

                    continue

                # We copy tensors synchronously as we do not want to accumulate
                # host tensors in memory.
                state_tensor.copy_(tensor, non_blocking=False)

                del tensor

                memo.add(state_tensor)

                progress_task.step()

    if state_dict:
        keys_to_remove = []

        # Ensure that we remove any shared tensor from `state_dict`.
        for key, tensor in state_dict.items():
            if tensor in memo:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del state_dict[key]

        if state_dict:
            missing_keys = list(state_dict.keys())

            s = ", ".join(missing_keys)

            errors.append(f"The following keys are missing in the checkpoint: {s}")

    if unexpected_keys:
        s = ", ".join(unexpected_keys)

        errors.append(f"The following keys in the checkpoint are unexpected: {s}")

    if errors:
        raise ValueError(" ".join(errors))


def get_converted_key(key: str, key_map: Mapping[str, str]) -> str:
    for pattern, replacement in key_map.items():
        if (converted_key := re.sub(pattern, replacement, key)) != key:
            return converted_key

    return key


def convert_checkpoint(
    checkpoint: dict[str, object], key_map: Mapping[str, str]
) -> dict[str, object]:
    """
    Converts a checkpoint.

    :param key_map: A map of regex patterns to update model keys.
    """
    converted_checkpoint = {}

    for key in checkpoint.keys():
        converted_key = get_converted_key(key, key_map)

        converted_checkpoint[converted_key] = checkpoint[key]

    return converted_checkpoint


def convert_fairseq_checkpoint(
    checkpoint: dict[str, object], key_map: Mapping[str, str]
) -> dict[str, object]:
    """
    Converts a fairseq checkpoint to fairseq2.

    :param checkpoint: The original fairseq checkpoint.
    :param key_map: A map of regex patterns to fairseq2 model keys.
    """
    fs2_checkpoint = convert_checkpoint(checkpoint, key_map)

    try:
        del fs2_checkpoint["encoder.version"]
    except KeyError:
        pass
    try:
        del fs2_checkpoint["decoder.version"]
    except KeyError:
        pass

    try:
        del fs2_checkpoint["encoder.embed_positions._float_tensor"]
    except KeyError:
        pass
    try:
        del fs2_checkpoint["decoder.embed_positions._float_tensor"]
    except KeyError:
        pass

    return fs2_checkpoint


def create_reverse_key_map(key_map: dict[str, str]) -> dict[str, str]:
    """Creates a reversed version of a regex-based key map."""
    reverse_key_map = {}

    for pattern, replacement in key_map.items():
        # Strip ^ from `pattern` if present.
        pattern_without_anchor = pattern[1:] if pattern.startswith("^") else pattern

        # Create a new pattern from the original replacement:
        #   1. Escape dots.
        #   2. Replace backreferences with capture groups.
        new_pattern = "^" + replacement.replace(".", r"\.").replace(r"\1", r"([0-9]+)")

        # Create a new replacement from the original pattern. Instead of string
        # manipulation, use a simpler approach. The key insight is that we need
        # \1 as a literal in the output string
        if "([0-9]+)" in pattern_without_anchor:
            # This is the literal representation we want in the final string.
            new_replacement = pattern_without_anchor.replace(r"([0-9]+)", r"\1")

            # Remove escaping from dots.
            new_replacement = new_replacement.replace(r"\.", ".")
        else:
            new_replacement = pattern_without_anchor.replace(r"\.", ".")

        reverse_key_map[new_pattern] = new_replacement

    return reverse_key_map
