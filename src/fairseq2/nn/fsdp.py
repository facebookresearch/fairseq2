# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Final, Optional, Protocol, Sequence, Set, final

import torch
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.nn import Module

from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.nn.utils.module import (
    infer_device,
    reset_non_persistent_buffers,
    reset_parameters,
    to_empty,
)
from fairseq2.typing import DataType, Device
from fairseq2.utils.version import torch_greater_or_equal

log = get_log_writer(__name__)


def to_fsdp(
    module: Module,
    gang: Gang,
    wrap_policy: Optional[FSDPWrapPolicy],
    *,
    ignored_modules: Optional[Sequence[Module]] = None,
    skip_init: bool = False,
    broadcast_state: bool = False,
    memory_policy: Optional[FSDPMemoryPolicy] = None,
    reshard_after_forward: bool = True,
    local_world_size: Optional[int] = None,
    mixed_precision_dtype: Optional[DataType] = None,
    fp32_reduce: bool = False,
) -> FSDP:
    """Wrap ``module`` with FSDP.

    :param module:
        The module to wrap.
    :param gang:
        The gang over which to shard ``module``.
    :param wrap_policy:
        The FSDP wrap policy to apply to ``module``. If ``None``, wraps only
        ``module`` itself.
    :param ignored_param_names:
        The ignored parameter names. Can contain regular expressions.
    :param skip_init:
        If ``True``, skips initializing the parameters and buffers moved from
        the meta device onto the device of ``gang``. Only relevant if ``module``
        resides on the meta device.
    :param broadcast_state:
        If ``True``, each FSDP module will broadcast its parameters and buffers
        from rank 0 to ensure that they are replicated across all processes.
    :param memory_policy:
        The policy to instruct FSDP when and how to allocate memory.
    :param reshard_after_forward:
        If ``True``, unshards the parameters before the forward pass and only
        reshards them after the backward pass.
    :param local_world_size:
        If not ``None``, enables hybrid sharding. ``gang`` will be split into
        sub-gangs each containing ``local_world_size`` number of consecutive
        processes. The model will be fully sharded within each sub-gang and
        will be replicated across sub-gangs.
    :param mixed_precision_dtype:
        If not ``None``, parameters, buffers, and gradients will use this data
        type during forward and backward passes. Outside forward and backward
        passes, the model will be kept in full precision.
    :param fp32_reduce:
        If ``True``, the gradients will be reduced in full precision. Only
        relevant if ``mixed_precision_dtype`` is not ``None``.
    """
    if local_world_size is not None:
        if local_world_size == 0:
            raise ValueError(
                f"`local_world_size` must be greater than 0, but is {local_world_size} instead."
            )

        if local_world_size > gang.size:
            raise ValueError(
                f"`local_world_size` must be less than or equal to `gang.size` ({gang.size}), but is {local_world_size} instead."
            )

        if gang.size % local_world_size != 0:
            raise ValueError(
                f"`gang.size` ({gang.size}) must be divisible by `local_world_size` ({local_world_size})."
            )

        # TODO(balioglu): Finish!
        raise NotImplementedError("`local_world_size` is not supported yet.")
    else:
        if reshard_after_forward:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

    if memory_policy is None:
        memory_policy = FSDP_STANDARD_MEMORY_POLICY

    param_init_fn = None

    module_device = infer_device(module)
    if module_device.type == "meta":
        if not torch_greater_or_equal(2, 1):
            log.warning("FSDP meta initialization is only supported on PyTorch 2.1.0 and later.")  # fmt: skip

            to_empty(module, gang.device)

            if not broadcast_state:
                reset_parameters(module)
        else:
            param_init_fn = FSDPParameterInitializer(gang.device, skip_init)

    if mixed_precision_dtype is None:
        mp = None
    elif mixed_precision_dtype == torch.float32:
        mp = None
    else:
        reduce_dtype = torch.float32 if fp32_reduce else mixed_precision_dtype

        mp = MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=None,
        )

    kwargs: Dict[str, Any] = {}

    # As of PyTorch 2.0, FSDP initialization fails in certain settings when an
    # empty `ignored_states` is specified (e.g. `sync_module_states` is set).
    if ignored_modules:
        kwargs["ignored_states"] = ignored_modules

    fsdp = FSDP(
        module,
        process_group=gang.as_process_group(),
        sharding_strategy=sharding_strategy,
        cpu_offload=CPUOffload() if memory_policy.cpu_offload else None,
        auto_wrap_policy=wrap_policy,
        backward_prefetch=memory_policy.backward_prefetch,
        mixed_precision=mp,
        param_init_fn=param_init_fn,
        device_id=gang.device,
        sync_module_states=broadcast_state,
        forward_prefetch=False,
        limit_all_gathers=memory_policy.limit_all_gathers,
        use_orig_params=True,
        **kwargs,
    )

    FSDP.set_state_dict_type(
        fsdp,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    )

    return fsdp


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


@final
@dataclass(frozen=True)
class FSDPMemoryPolicy:
    """Specifies the device memory usage policy of an FSDP module."""

    backward_prefetch: Optional[BackwardPrefetch]
    """The backward prefetch mode for all-gathers. For more information, check
    out the same named parameter of :class:`FSDP`."""

    limit_all_gathers: bool
    """If ``True``, FSDP explicitly synchronizes the CPU thread to ensure GPU
    memory use from only two consecutive FSDP instances. For more information,
    check out the same named parameter of :class:`FSDP`."""

    cpu_offload: bool
    """If ``True``, FSDP offloads parameters not involved in computation to CPU.
    For more information, check out :class:`CPUOffload`."""


FSDP_STANDARD_MEMORY_POLICY: Final = FSDPMemoryPolicy(
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    limit_all_gathers=True,
    cpu_offload=False,
)
"""Enables backward prefetching."""


FSDP_LOW_MEMORY_POLICY: Final = FSDPMemoryPolicy(
    backward_prefetch=BackwardPrefetch.BACKWARD_POST,
    limit_all_gathers=True,
    cpu_offload=False,
)
"""Enables backward prefetching with low-memory pressure."""


FSDP_VERY_LOW_MEMORY_POLICY: Final = FSDPMemoryPolicy(
    backward_prefetch=None,
    limit_all_gathers=True,
    cpu_offload=True,
)
"""Disables communication and computation overlap and offloads parameters to CPU."""


@final
class FSDPParameterInitializer:
    """Initializes the parameters and buffers of an FSDP module.

    This is a convenience callable to pass to the ``param_init_fn`` parameter of
    :class:`FSDP`. It moves the parameters and buffers residing on a meta device
    onto ``device`` and initializes them.

    Usage:

    >>> model = MyModel(..., device=Device("meta"))
    >>>
    >>> fsdp_model = FullyShardedDataParallel(
    ...     ..., param_init_fn=FSDPParameterInitializer(Device("cuda:0"))
    ... )
    """

    _module_memo: Set[Module]
    _memo: Dict[Tensor, Tensor]
    _device: Device
    _skip_init: bool

    def __init__(self, device: Device, skip_init: bool = False) -> None:
        """
        :param device:
            The device onto which to move the parameters and buffers.
        :param skip_init:
            If ``True``, skips initializing the parameters and buffers after
            moving them onto ``device``. The non-persistent buffers are always
            initialized regardless of ``skip_init``.
        """
        self._module_memo = set()
        self._memo = {}
        self._device = device
        self._skip_init = skip_init

    def __call__(self, module: Module) -> None:
        """
        :param module:
            An FSDP module or submodule.
        """
        if module in self._module_memo:
            return

        to_empty(module, self._device, recurse=False, memo=self._memo)

        if not self._skip_init:
            reset_parameters(module, recurse=False)
        else:
            # Non-persistent buffers are never part of module's state, so we
            # have to initialize them even with `skip_init`.
            reset_non_persistent_buffers(module, recurse=False)

        self._module_memo.add(module)
