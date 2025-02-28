# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, Protocol, TypeAlias, final

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._shard.sharded_tensor import ShardedTensor
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

from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.nn.data_parallel._error import DistributedSetupError
from fairseq2.nn.utils.module import (
    apply_to_parameters,
    infer_device,
    reset_non_persistent_buffers,
    reset_parameters,
    to_empty,
)
from fairseq2.typing import DataType, Device

FsdpGranularity: TypeAlias = Literal["layer", "stack", "model"]


class FsdpWrapper(Protocol):
    def __call__(
        self, module: Module, reshard_after_forward: bool | None = None
    ) -> Module: ...


class FsdpApplier(Protocol):
    def __call__(
        self, module: Module, granularity: FsdpGranularity, wrapper: FsdpWrapper
    ) -> Module: ...


def to_fsdp(
    module: Module,
    gangs: Gangs,
    applier: FsdpApplier,
    *,
    granularity: FsdpGranularity = "layer",
    broadcast_state: bool = False,
    cpu_offload: bool = False,
    reshard_after_forward: bool = True,
    mixed_precision_dtype: DataType | None = None,
    fp32_reduce: bool = False,
) -> FSDP:
    """Wrap ``module`` with FSDP.

    :param module: The module to wrap.
    :param gangs: The gangs over which to shard ``module``.
    :param applier: The callable to apply FSDP to ``module``.
    :param ignored_param_names: The ignored parameter names. Can contain regular
        expressions.
    :param broadcast_state: If ``True``, each FSDP module will broadcast its
        parameters and buffers from rank 0 to ensure that they are replicated
        across all processes.
    :param reshard_after_forward: If ``True``, unshards the parameters before
        the forward pass and only reshards them after the backward pass.
    :param mixed_precision_dtype: If not ``None``, parameters, buffers, and
        gradients will use this data type during forward and backward passes.
        Outside forward and backward passes, the model will be kept in full
        precision.
    :param fp32_reduce: If ``True``, the gradients will be reduced in full
        precision. Only relevant if ``mixed_precision_dtype`` is not ``None``.
    """
    if gangs.sdp.size == 1:
        raise NotSupportedError(
            "FSDP does not support non-sharded data parallelism. Please use DDP instead."
        )

    process_group: ProcessGroup | tuple[ProcessGroup, ProcessGroup]

    # Determine the sharding strategy.
    if gangs.rdp.size > 1:
        if reshard_after_forward:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

        try:
            process_group = (gangs.sdp.as_process_group(), gangs.rdp.as_process_group())
        except NotSupportedError:
            raise DistributedSetupError(
                "The specified data parallel gang does not support conversion to a process group."
            ) from None
    else:
        if reshard_after_forward:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        try:
            process_group = gangs.sdp.as_process_group()
        except NotSupportedError:
            raise DistributedSetupError(
                "The specified data parallel gang does not support conversion to a process group."
            ) from None

    # Set up parameter initialization.
    try:
        module_device = infer_device(module)
    except ValueError as ex:
        raise DistributedSetupError(
            "The device of `module` is not valid. See the nested exception for details."
        ) from ex

    if module_device.type != "meta":
        param_initializer = None
    else:
        if broadcast_state:
            if gangs.dp.rank == 0:
                raise DistributedSetupError(
                    "`broadcast_state` is set, but the coordinator process (i.e. rank 0) is on a meta device."
                )

            skip_init = True
        else:
            skip_init = False

        param_initializer = FsdpParameterInitializer(gangs.dp.device, skip_init)

    # Set up the data types for mixed precision training.
    if mixed_precision_dtype is None:
        mp = None
    elif mixed_precision_dtype == torch.float32:
        mp = None
    else:
        reduce_dtype = torch.float32 if fp32_reduce else mixed_precision_dtype

        mp = MixedPrecision(mixed_precision_dtype, reduce_dtype, buffer_dtype=None)

    if not cpu_offload:
        cpu_offload_ = None
    else:
        cpu_offload_ = CPUOffload()

    def wrap(module: Module, reshard_after_forward: bool | None = None) -> FSDP:
        if reshard_after_forward is None:
            sharding_strategy_ = sharding_strategy
        else:
            if reshard_after_forward:
                if gangs.rdp.size == 1:
                    sharding_strategy_ = ShardingStrategy.SHARD_GRAD_OP
                else:
                    sharding_strategy_ = ShardingStrategy._HYBRID_SHARD_ZERO2
            else:
                if gangs.rdp.size == 1:
                    sharding_strategy_ = ShardingStrategy.FULL_SHARD
                else:
                    sharding_strategy_ = ShardingStrategy.HYBRID_SHARD

        try:
            return FSDP(
                module,
                process_group=process_group,
                sharding_strategy=sharding_strategy_,
                cpu_offload=cpu_offload_,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                mixed_precision=mp,
                param_init_fn=param_initializer,
                device_id=gangs.dp.device,
                sync_module_states=broadcast_state,
                forward_prefetch=False,
                limit_all_gathers=True,
                use_orig_params=True,
            )
        except (RuntimeError, ValueError) as ex:
            raise DistributedSetupError(
                "FSDP cannot be initialized. See the nested exception for details."
            ) from ex

    module = applier(module, granularity, wrap)

    if not isinstance(module, FSDP):
        module = wrap(module, reshard_after_forward=False)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", message=r".*FSDP\.state_dict_type\(\) and FSDP\.set_state_dict_type\(\) are being deprecated.*"  # fmt: skip
        )

        FSDP.set_state_dict_type(
            module,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=False),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=False),
        )

    return module


@final
class FsdpParameterInitializer:
    """Initializes the parameters and buffers of an FSDP module.

    This is a convenience callable to pass to the ``param_init_fn`` parameter of
    :class:`FSDP`. It moves the parameters and buffers residing on a meta device
    onto ``device`` and initializes them.

    Usage:

    >>> model = MyModel(..., device=Device("meta"))
    >>>
    >>> fsdp_model = FullyShardedDataParallel(
    ...     ..., param_init_fn=FsdpParameterInitializer(Device("cuda:0"))
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


def fsdp_local_state_dict(module: FSDP) -> dict[str, object]:
    state_dict: dict[str, object] = {}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", message=r".*`_get_pg_default_device` will be deprecated.*"  # fmt: skip
        )
        warnings.filterwarnings(
            action="ignore", message=r".*Please use DTensor instead.*"
        )

        sdp_rank = dist.get_rank(module.process_group)  # sharded

        for name, item in module.state_dict().items():
            if isinstance(item, ShardedTensor):
                local_shards = item.local_shards()
                if not local_shards:
                    continue  # means the tensor is sharded unevenly.

                state_dict[name] = item.local_tensor().detach()
            # Save replicated items only on the first intra-node (i.e. sharded)
            # gang.
            elif sdp_rank == 0:
                if isinstance(item, Tensor):
                    item = item.detach()

                state_dict[name] = item

    return state_dict


@contextmanager
def fsdp_summon_full_parameters(module: FSDP) -> Iterator[None]:
    """Unshard the parameters of ``module`` and use the non-FSDP forward method."""
    mp = module.mixed_precision or MixedPrecision()

    # This is ugly, but our only option. We monkey-patch FSDP modules to
    # replace their `forward` methods with the wrapped `forward` methods.
    # Otherwise, FSDP fails to shard parameters at the end of the call.
    def disable_fsdp_forward(module_: Module) -> None:
        for m in module_.modules():
            if isinstance(m, FSDP):
                m._fs2_backup_forward = m.forward  # type: ignore[assignment]

                m.forward = m.module.forward  # type: ignore[method-assign]

    def enable_fsdp_forward(module_: Module) -> None:
        for m in module_.modules():
            if isinstance(m, FSDP):
                m.forward = m._fs2_backup_forward  # type: ignore[method-assign]

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
