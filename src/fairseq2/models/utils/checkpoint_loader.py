# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Callable, Dict, Mapping, NoReturn, Optional, Protocol, Union

import torch
from torch import Tensor
from typing_extensions import TypeAlias

from fairseq2.data.typing import PathLike
from fairseq2.typing import Device

MapLocation: TypeAlias = Optional[
    Union[Callable[[Tensor, str], Tensor], Device, str, Dict[str, str]]
]


class CheckpointConverter(Protocol):
    """Converts a checkpoint to make it compatible with fairseq2."""

    def __call__(self, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        :param checkpoint:
            The checkpoint to convert.

        :returns:
            A converted checkpoint that is compatible with fairseq2.
        """


def load_checkpoint(
    pathname: PathLike,
    model_name: str,
    checkpoint_name: Optional[str] = None,
    map_location: MapLocation = None,
    restrict: bool = False,
    converter: Optional[CheckpointConverter] = None,
) -> Mapping[str, Any]:
    """Load the checkpoint stored in ``pathname``.

    :param pathname:
        The pathname of the checkpoint.
    :param model_name:
        The name of the associated model.
    :param checkpoint_name:
        The name of the checkpoint.
    :param map_location:
        Same as the ``map_location`` parameter of :meth:`torch.load`.
    :param restrict:
        If ``True``, restricts the Python unpickler to load only tensors,
        primitive types, and dictionaries.
    :param converter:
        The callable to which the loaded checkpoint will be passed for further
        processing. Typically used to upgrade legacy checkpoints.

    :returns:
        The loaded checkpoint.
    """

    def raise_error(cause: Exception) -> NoReturn:
        if not checkpoint_name:
            display_name = f"checkpoint of the model '{model_name}'"
        else:
            display_name = f"'{checkpoint_name}' checkpoint of the model '{model_name}'"

        raise RuntimeError(
            f"The load of the {display_name} has failed. See nested exception for details."
        ) from cause

    try:
        checkpoint: Mapping[str, Any] = torch.load(
            str(pathname), map_location, weights_only=restrict
        )
    except IOError as ex:
        raise_error(ex)

    if converter is not None:
        try:
            checkpoint = converter(checkpoint)
        except (KeyError, ValueError) as ex:
            raise_error(ex)

    return checkpoint


def upgrade_fairseq_checkpoint(
    checkpoint: Mapping[str, Any], key_map: Mapping[str, str]
) -> Dict[str, Any]:
    """Upgrade a fairseq checkpoint.

    :param checkpoint:
        The original fairseq checkpoint.
    :param key_map:
        A map of regex patterns to replacement strings to modify the model keys.

    :returns:
        A converted checkpoint that is compatible with fairseq2.
    """
    old_state_dict = checkpoint["model"]
    new_state_dict = {}

    def get_new_key(old_key: str) -> str:
        for old_pattern, replacement in key_map.items():
            if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                return new_key

        return old_key

    # Convert module keys from fairseq to fairseq2.
    for old_key in old_state_dict.keys():
        new_key = get_new_key(old_key)

        new_state_dict[new_key] = old_state_dict[old_key]

    # Use the built-in version attribute of `torch.nn.Module`.
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
