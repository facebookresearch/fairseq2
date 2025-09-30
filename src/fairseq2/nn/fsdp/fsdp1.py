# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeAlias

import torch
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
from torch.nn import Module, Parameter, SyncBatchNorm

from fairseq2.data_type import DataType
from fairseq2.error import NotSupportedError, OperationalError
from fairseq2.gang import GangError, Gangs
from fairseq2.nn.fsdp.common import FSDPApplier, FSDPParameterInitializer
from fairseq2.nn.utils.module import (
    apply_to_parameters,
    load_state_dict,
    maybe_infer_device,
)

# Suppress excessively noisy FSDP log messages that are wrongfully output at
# the warning level when TORCH_DISTRIBUTED_DEBUG is set to INFO.
logging.getLogger("torch.distributed.fsdp._runtime_utils").setLevel(logging.ERROR)


FSDP1Module: TypeAlias = FSDP


def to_fsdp1(
    module: Module,
    gangs: Gangs,
    applier: FSDPApplier,
    *,
    mixed_precision_dtype: DataType | None = None,
    fp32_reduce: bool = False,
    broadcast_state: bool = False,
    skip_init: bool = False,
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
) -> FSDP1Module:
    if gangs.sdp.size == 1:
        raise NotSupportedError(
            "FSDP1 does not support non-sharded data parallelism. Use DDP instead."
        )

    process_group: ProcessGroup | tuple[ProcessGroup, ProcessGroup]

    # Determine the sharding strategy.
    if gangs.rdp.size > 1:
        if reshard_after_forward:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

        process_group = (gangs.sdp.as_process_group(), gangs.rdp.as_process_group())
    else:
        if reshard_after_forward:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        process_group = gangs.sdp.as_process_group()

    # Set up parameter initialization.
    device = maybe_infer_device(module)
    if device is None:
        raise ValueError(
            "All parameters and buffers of `module` must be on the same device."
        )

    if device.type != "meta":
        param_initializer = None
    else:
        if broadcast_state:
            if gangs.dp.rank == 0:
                raise ValueError(
                    "Coordinator process (i.e. rank 0) must be on a real device when `broadcast_state` is set."
                )

            skip_init = True

        param_initializer = FSDPParameterInitializer(gangs.dp.device, skip_init)

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

    def wrap(module: Module, reshard_after_forward: bool | None = None) -> FSDP1Module:
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
            return FSDP1Module(
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
        except RuntimeError as ex:
            raise GangError("FSDP1 parameter synchronization failed.") from ex

    module = applier(module, wrap)

    if not isinstance(module, FSDP1Module):
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

    # FSDP does not broadcast buffers during training, so ensure that batch
    # normalization works.
    dp_process_group = gangs.dp.as_process_group()

    SyncBatchNorm.convert_sync_batchnorm(module, dp_process_group)

    return module


def fsdp1_local_state_dict(module: FSDP1Module) -> dict[str, object]:
    state_dict: dict[str, object] = {}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", message=r".*`_get_pg_default_device` will be deprecated.*"  # fmt: skip
        )
        warnings.filterwarnings(
            action="ignore", message=r".*Please use DTensor instead.*"
        )

        for name, item in module.state_dict().items():
            if isinstance(item, ShardedTensor):
                local_shards = item.local_shards()
                if not local_shards:
                    continue  # means the tensor is sharded unevenly.

                item = item.local_tensor().detach()
            elif isinstance(item, Tensor):
                item = item.detach()

            state_dict[name] = item

    return state_dict


def fsdp1_load_local_state_dict(
    module: FSDP1Module, state_dict: dict[str, object]
) -> None:
    state_dict = dict(state_dict)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", message=r".*`_get_pg_default_device` will be deprecated.*"  # fmt: skip
        )
        warnings.filterwarnings(
            action="ignore", message=r".*Please use DTensor instead.*"
        )

        for key, value in module.state_dict().items():
            if isinstance(value, ShardedTensor):
                input_value = state_dict.get(key)
                if isinstance(input_value, Tensor):
                    value.local_tensor().detach().copy_(input_value)

                    state_dict[key] = value

        load_state_dict(module, state_dict)


@contextmanager
def fsdp1_summon_full_parameters(module: FSDP1Module) -> Iterator[None]:
    FORWARD_BACKUP_KEY = "__fs2_forward_backup__"

    mp = module.mixed_precision or MixedPrecision()

    # FSDP does not support calling `module.forward()` when the parameters are
    # summoned. As a workaround, we monkey-patch FSDP modules to replace their
    # `forward()` methods with the wrapped `__call__()` methods.
    def disable_fsdp_forward(module: Module) -> None:
        for m in module.modules():
            if isinstance(m, FSDP1Module):
                setattr(m, FORWARD_BACKUP_KEY, m.forward)

                setattr(m, "forward", m.module.__call__)

    def enable_fsdp_forward(module: Module) -> None:
        for m in module.modules():
            if isinstance(m, FSDP1Module):
                fwd = getattr(m, FORWARD_BACKUP_KEY)

                setattr(m, "forward", fwd)

                delattr(m, FORWARD_BACKUP_KEY)

    def maybe_cast_dtype(t: Tensor) -> Tensor:
        dtype = mp.param_dtype if isinstance(t, Parameter) else mp.buffer_dtype

        if dtype is None:
            return t

        return t.to(dtype)

    try:
        with FSDP.summon_full_params(module, writeback=False):
            disable_fsdp_forward(module)

            apply_to_parameters(module, maybe_cast_dtype)

            try:
                yield
            finally:
                enable_fsdp_forward(module)
    except RuntimeError as ex:
        raise OperationalError("FSDP1 summon operation failed.") from ex
