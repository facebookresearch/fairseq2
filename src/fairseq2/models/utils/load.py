# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Mapping, NoReturn, Optional

import torch
from torch.serialization import MAP_LOCATION
from typing_extensions import TypeAlias

from fairseq2.data.typing import PathLike

CheckpointUpgrader: TypeAlias = Callable[[Mapping[str, Any]], Dict[str, Any]]


def load_checkpoint(
    pathname: PathLike,
    model_name: str,
    map_location: MAP_LOCATION = None,
    restrict: bool = False,
    checkpoint_upgrader: Optional[CheckpointUpgrader] = None,
) -> Dict[str, Any]:
    """Load the checkpoint stored in ``pathname``.

    :param pathname:
        The pathname of the checkpoint.
    :param model_name:
        The name of the associated model.
    :param map_location:
        Same as the ``map_location`` parameter of :meth:`torch.load`.
    :param restrict:
        If ``True``, the Python unpickler will be restricted to loading only
        tensors, primitive types, and dictionaries.
    :param model_upgrader:
        The callable to which the loaded checkpoint will be passed for further
        processing. Typically used to upgrade legacy checkpoints.
    """

    def raise_error(cause: Exception) -> NoReturn:
        raise RuntimeError(
            f"The {model_name} checkpoint cannot be loaded. Please file a bug report."
        ) from cause

    try:
        checkpoint: Dict[str, Any] = torch.load(
            str(pathname), map_location, weights_only=restrict
        )
    except IOError as ex:
        raise_error(ex)

    if checkpoint_upgrader is not None:
        try:
            checkpoint = checkpoint_upgrader(checkpoint)
        except (KeyError, ValueError) as ex:
            raise_error(ex)

    return checkpoint
