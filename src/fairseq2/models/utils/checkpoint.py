# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor
from torch.nn import Module


def load_checkpoint(model: Module, checkpoint: Iterable[tuple[str, Tensor]]) -> None:
    errors = []

    unexpected_keys = []

    memo = set()

    # To determine shared tensors we have to preserve the identity of the
    # parameters and buffers via `keep_vars`.
    state_dict = model.state_dict(keep_vars=True)

    with torch.no_grad():
        for key, tensor in checkpoint:
            try:
                state_tensor = state_dict.pop(key)
            except KeyError:
                unexpected_keys.append(key)

                continue

            if tensor.shape != state_tensor.shape:
                errors.append(
                    f"`{key}` has a shape of {tuple(tensor.shape)} in the checkpoint, but has a shape of {tuple(state_tensor.shape)} in the model."
                )

                continue

            state_tensor.copy_(tensor, non_blocking=True)

            memo.add(state_tensor)

    if state_dict:
        keys_to_remove = []

        # Ensure that we remove any shared tensors from `state_dict`.
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


def convert_checkpoint(
    checkpoint: dict[str, object], key_map: Mapping[str, str]
) -> dict[str, object]:
    """Convert a checkpoint.

    :param key_map: A map of regex patterns to update model keys.
    """
    converted_checkpoint = {}

    def get_converted_key(key: str) -> str:
        for pattern, replacement in key_map.items():
            if (converted_key := re.sub(pattern, replacement, key)) != key:
                return converted_key

        return key

    for key in checkpoint.keys():
        converted_key = get_converted_key(key)

        converted_checkpoint[converted_key] = checkpoint[key]

    return converted_checkpoint


def convert_fairseq_checkpoint(
    checkpoint: dict[str, object], key_map: Mapping[str, str]
) -> dict[str, object]:
    """Convert a fairseq checkpoint to fairseq2.

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
