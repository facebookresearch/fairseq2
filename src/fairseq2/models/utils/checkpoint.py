# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from collections.abc import Mapping


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
