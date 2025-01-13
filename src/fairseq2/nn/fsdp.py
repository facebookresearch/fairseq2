# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Protocol, final
from warnings import catch_warnings

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.nn import Module, Parameter

from fairseq2.gang import Gang, setup_hybrid_fsdp_gangs
from fairseq2.logging import log
from fairseq2.nn.utils.module import (
    apply_to_parameters,
    infer_device,
    reset_non_persistent_buffers,
    reset_parameters,
    to_empty,
)
from fairseq2.typing import DataType, Device


def to_fsdp(
    module: Module,
    gang: Gang,
    wrap_policy: FSDPWrapPolicy | None,
    *,
    ignored_modules: Sequence[Module] | None = None,
    skip_init: bool = False,
    broadcast_state: bool = False,
    memory_policy: FSDPMemoryPolicy | None = None,
    reshard_after_forward: bool = True,
    local_world_size: int | None = None,
    mixed_precision_dtype: DataType | None = None,
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
        Not used.
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
    process_group: ProcessGroup | tuple[ProcessGroup, ProcessGroup] | None = None

    if local_world_size is not None:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD

        sharding_gang, replication_gang = setup_hybrid_fsdp_gangs(
            gang, local_world_size
        )

        process_group = (
            sharding_gang.as_process_group(),
            replication_gang.as_process_group(),
        )
    else:
        if reshard_after_forward:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        process_group = gang.as_process_group()

    if memory_policy is None:
        memory_policy = FSDP_STANDARD_MEMORY_POLICY

    if skip_init:
        log.warning("`skip_init` parameter has no effect and will be removed in a future release.")  # fmt: skip

    param_init_fn = None

    try:
        module_device = infer_device(module)
    except ValueError as ex:
        raise ValueError(
            "The device of `module` is not valid. See the nested exception for details."
        ) from ex

    if module_device.type == "meta":
        if gang.rank == 0:
            skip_init = not broadcast_state
        else:
            skip_init = True

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

    kwargs: dict[str, Any] = {}

    # As of PyTorch 2.0, FSDP initialization fails in certain settings when an
    # empty `ignored_states` is specified (e.g. `sync_module_states` is set).
    if ignored_modules:
        kwargs["ignored_states"] = ignored_modules

    fsdp = FSDP(
        module,
        process_group=process_group,
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

    with catch_warnings():
        warnings.simplefilter("ignore")  # Suppress noisy FSDP warnings.

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

    backward_prefetch: BackwardPrefetch | None
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

    _module_memo: set[Module]
    _memo: dict[Tensor, Tensor]
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


# mypy: disable-error-code="method-assign"


@contextmanager
def summon_fsdp_for_validation(module: Module) -> Iterator[None]:
    """Unshard the parameters of ``module`` and use the non-FSDP forward method."""
    if not isinstance(module, FSDP):
        yield
    else:
        mp = module.mixed_precision or MixedPrecision()

        # This is ugly, but our only option. We monkey-patch FSDP modules to
        # replace their `forward` methods with the wrapped `forward` methods.
        # Otherwise, FSDP fails to shard parameters at the end of the call.
        def disable_fsdp_forward(module_: Module) -> None:
            for m in module_.modules():
                if isinstance(m, FSDP):
                    m._fs2_backup_forward = m.forward  # type: ignore[assignment]

                    m.forward = m.module.forward

        def enable_fsdp_forward(module_: Module) -> None:
            for m in module_.modules():
                if isinstance(m, FSDP):
                    m.forward = m._fs2_backup_forward

                    del m._fs2_backup_forward

        def maybe_cast_dtype(t: Tensor) -> Tensor:
            dtype = mp.param_dtype if isinstance(t, Parameter) else mp.buffer_dtype

            if dtype is None:
                return t

            return t.to(dtype)

        with FSDP.summon_full_params(module, writeback=False):
            disable_fsdp_forward(module)

            apply_to_parameters(module, maybe_cast_dtype)

            try:
                yield
            finally:
                enable_fsdp_forward(module)
