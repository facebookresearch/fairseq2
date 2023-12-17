# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterable, Optional, Protocol, Sequence

from torch import Tensor
from torch.nn import Module, Parameter

from fairseq2.nn.utils.module import reset_parameters, select_parameters, to_empty
from fairseq2.typing import Device


class FSDPWrapPolicy(Protocol):
    """Represents an FSDP wrap policy."""

    def __call__(self, module: Module, recurse: bool, non_wrapped_numel: int) -> bool:
        """
        :param module:
            The module to apply the policy to.
        :param recurse:
            If ``False``, the return value specifies whether ``module`` should
            have FSDP applied; if ``True``, the return value specifies whether
            the traversal should continue into the module's subtree.
        :param non_wrapped_numel:
            The number of elements that have not yet been wrapped.

        :returns:
            See the description of the ``recurse`` parameter.
        """


def get_ignored_parameters(
    module: Module, names: Optional[Sequence[str]]
) -> Optional[Iterable[Parameter]]:
    """Get the list of parameters that should be ignored by FSDP.

    :param module:
        The module to be wrapped by FSDP.
    :param names:
        The ignored parameter names, can contain regular expressions.
    """
    if names is None:
        return None

    return (p for _, p in select_parameters(module, names))


class FSDPParameterInitializer:
    """Initializes the parameters and buffers of an FSDP module.

    This is a convenience callable to pass to the ``param_init_fn`` parameter of
    the FSDP constructor. It moves the parameters and buffers residing on a meta
    device to ``device`` and initializes them.

    Usage:

    >>> model = MyModel(..., device=Device("meta"))
    >>>
    >>> fsdp_model = FullyShardedDataParallel(
    ...     ..., param_init_fn=FSDPParameterInitializer(Device("cuda:0"))
    ... )
    """

    memo: Dict[Tensor, Tensor]
    device: Device

    def __init__(self, device: Device) -> None:
        """
        :param device:
            The device on which to initialize the parameters and buffers.
        """
        self.memo = {}
        self.device = device

    def __call__(self, module: Module) -> None:
        """
        :param module:
            An FSDP module or submodule.
        """
        to_empty(module, self.device, recurse=False, memo=self.memo)

        reset_parameters(module, recurse=False)
