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

ParameterUpgrader: TypeAlias = Callable[[Mapping[str, Any]], Dict[str, Any]]


def load_parameters(
    pathname: PathLike,
    model_name: str,
    map_location: MAP_LOCATION = None,
    weights_only: bool = False,
    param_upgrader: Optional[ParameterUpgrader] = None,
) -> Dict[str, Any]:
    """Loads the parameters of the model stored in ``pathname``.

    :param pathname:
        The pathname of the model file.
    :param model_name:
        The model name to display in error messages.
    :param map_location:
        Same as the ``map_location`` parameter of :meth:`torch.load`.
    :param weights_only:
        If ``True``, the Python unpickler will be restricted to loading only
        tensors, primitive types, and dictionaries.
    :param param_upgrader:
        The callable to which the loaded parameters will be passed for further
        processing. Typically used for loading models from legacy files.
    """

    def raise_error(cause: Exception) -> NoReturn:
        raise RuntimeError(
            f"The {model_name} model parameters cannot be loaded. Please file a bug report."
        ) from cause

    try:
        params: Dict[str, Any] = torch.load(
            str(pathname), map_location, weights_only=weights_only
        )
    except IOError as ex:
        raise_error(ex)

    if param_upgrader is not None:
        try:
            params = param_upgrader(params)
        except (KeyError, ValueError) as ex:
            raise_error(ex)

    return params
