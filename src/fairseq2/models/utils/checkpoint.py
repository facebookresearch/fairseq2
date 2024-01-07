# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Union

import torch
from torch import Tensor
from typing_extensions import TypeAlias

from fairseq2.data.typing import PathLike
from fairseq2.typing import Device
from fairseq2.utils.version import _is_pt21_or_greater

MapLocation: TypeAlias = Optional[
    Union[Callable[[Tensor, str], Tensor], Device, str, Dict[str, str]]
]


class CheckpointConverter(Protocol):
    """Converts checkpoints to fairseq2."""

    def __call__(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param checkpoint:
            The checkpoint to convert.

        :returns:
            A converted checkpoint that is compatible with fairseq2.
        """


def load_checkpoint(
    pathname: PathLike,
    *,
    map_location: MapLocation = None,
    restrict: bool = False,
    converter: Optional[CheckpointConverter] = None,
) -> Dict[str, Any]:
    """Load the checkpoint stored in ``pathname``.

    :param pathname:
        The pathname of the checkpoint.
    :param map_location:
        Same as the ``map_location`` parameter of :meth:`torch.load`.
    :param restrict:
        If ``True``, restricts the Python unpickler to load only tensors,
        primitive types, and dictionaries.
    :param converter:
        The converter to which the loaded checkpoint will be passed for further
        processing.

    :returns:
        The loaded checkpoint.
    """
    with warnings.catch_warnings():
        # Suppress the noisy deprecated `TypedStorage` warning.
        warnings.simplefilter("ignore")

        kwargs = {}

        if _is_pt21_or_greater():
            kwargs["mmap"] = True

        checkpoint: Dict[str, Any] = torch.load(
            str(pathname), map_location, weights_only=restrict, **kwargs
        )

    if converter is not None:
        checkpoint = converter(checkpoint)

    return checkpoint


def convert_model_state_dict(
    state_dict: Dict[str, Any], key_map: Mapping[str, str]
) -> Dict[str, Any]:
    """Convert a model state dictionary to fairseq2.

    :param state_dict:
        The original model state dictionary.
    :param key_map:
        A map of regex patterns to fairseq2 model keys.

    :returns:
        A converted model state dictionary that is compatible with fairseq2.
    """
    new_state_dict = {}

    def get_new_key(old_key: str) -> str:
        for old_pattern, replacement in key_map.items():
            if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                return new_key

        return old_key

    # Convert module keys from fairseq to fairseq2.
    for old_key in state_dict.keys():
        new_key = get_new_key(old_key)

        new_state_dict[new_key] = state_dict[old_key]

    return new_state_dict


def convert_fairseq_checkpoint(
    checkpoint: Dict[str, Any], key_map: Mapping[str, str]
) -> Dict[str, Any]:
    """Convert a fairseq checkpoint to fairseq2.

    :param checkpoint:
        The original fairseq checkpoint.
    :param key_map:
        A map of regex patterns to fairseq2 model keys.

    :returns:
        A converted checkpoint that is compatible with fairseq2.
    """
    old_state_dict = checkpoint["model"]

    new_state_dict = convert_model_state_dict(old_state_dict, key_map)

    # We use the built-in version attribute of `torch.nn.Module`.
    try:
        del new_state_dict["encoder.version"]
    except KeyError:
        pass
    try:
        del new_state_dict["decoder.version"]
    except KeyError:
        pass

    try:
        del new_state_dict["encoder.embed_positions._float_tensor"]
    except KeyError:
        pass
    try:
        del new_state_dict["decoder.embed_positions._float_tensor"]
    except KeyError:
        pass

    return {"model": new_state_dict}
