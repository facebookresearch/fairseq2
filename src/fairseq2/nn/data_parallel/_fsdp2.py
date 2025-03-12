# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeAlias, cast, final

import torch
from torch import Tensor
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor
from torch.nn import Module, Parameter, SyncBatchNorm

from fairseq2.error import InvalidOperationError, NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.nn.data_parallel._common import (
    FsdpApplier,
    FsdpGranularity,
    FsdpParameterInitializer,
)
from fairseq2.nn.data_parallel._error import DistributedSetupError
from fairseq2.nn.utils.module import apply_to_parameters, broadcast_module, infer_device
from fairseq2.typing import DataType

# Suppress yet another non-actionable FSDP warning as it originates from within
# PyTorch itself.
warnings.filterwarnings(
    action="ignore", message=r".*Found a non-scalar tensor with numel=1 and ndim!=0, we are implicitly creating a replicated DTensor for it.*"  # fmt: skip
)


if TYPE_CHECKING:

    @final
    class Fsdp2Module(Module):
        def unshard(self) -> None: ...

        def reshard(self) -> None: ...

        def set_requires_gradient_sync(self, value: bool) -> None: ...

else:
    Fsdp2Module: TypeAlias = FSDPModule


def to_fsdp2(
    module: Module,
    gangs: Gangs,
    applier: FsdpApplier,
    *,
    granularity: FsdpGranularity = "layer",
    mixed_precision_dtype: DataType | None = None,
    fp32_reduce: bool = False,
    broadcast_state: bool = False,
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
) -> Fsdp2Module:
    if gangs.sdp.size == 1:
        raise NotSupportedError(
            "FSDP2 does not support non-sharded data parallelism. Please use DDP instead."
        )

    dp_gang = gangs.dp

    # Determine the sharding strategy.
    if gangs.rdp.size > 1:
        try:
            process_groups = [
                gangs.rdp.as_process_group(),
                gangs.sdp.as_process_group(),
            ]
        except NotSupportedError:
            raise DistributedSetupError(
                "The specified data parallel gang does not support conversion to a process group."
            ) from None

        mesh = torch.arange(dp_gang.size).view(gangs.rdp.size, gangs.sdp.size)

        device_mesh = DeviceMesh.from_group(
            process_groups, dp_gang.device.type, mesh, mesh_dim_names=("inter", "intra")
        )
    else:
        try:
            process_group = gangs.sdp.as_process_group()
        except NotSupportedError:
            raise DistributedSetupError(
                "The specified data parallel gang does not support conversion to a process group."
            ) from None

        device_mesh = DeviceMesh.from_group(process_group, dp_gang.device.type)

    # FSDP2 does not broadcast buffers during training, so ensure that batch
    # normalization works.
    dp_process_group = dp_gang.as_process_group()

    SyncBatchNorm.convert_sync_batchnorm(module, dp_process_group)

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
            if dp_gang.rank == 0:
                raise DistributedSetupError(
                    "`broadcast_state` is set, but the coordinator process (i.e. rank 0) is on a meta device."
                )

            skip_init = True
        else:
            skip_init = False

        param_initializer = FsdpParameterInitializer(dp_gang.device, skip_init)

    # Set up the data types for mixed precision training.
    if mixed_precision_dtype is None:
        mp = MixedPrecisionPolicy()
    elif mixed_precision_dtype == torch.float32:
        mp = MixedPrecisionPolicy()
    else:
        reduce_dtype = torch.float32 if fp32_reduce else mixed_precision_dtype

        mp = MixedPrecisionPolicy(mixed_precision_dtype, reduce_dtype)

    if not cpu_offload:
        offload_policy = OffloadPolicy()
    else:
        offload_policy = CPUOffloadPolicy()

    # Shard the module.
    reshard_after_forward_ = reshard_after_forward

    sharded_modules: set[Module] = set()

    def wrap(module: Module, reshard_after_forward: bool | None = None) -> Fsdp2Module:
        if reshard_after_forward is None:
            reshard_after_forward = reshard_after_forward_

        if param_initializer is not None:
            param_initializer(module)

        if broadcast_state:
            broadcast_module(module, dp_gang, skip_modules=sharded_modules)

        fully_shard(
            module,
            mesh=device_mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp,
            offload_policy=offload_policy,
        )

        sharded_modules.add(module)

        return cast(Fsdp2Module, module)

    try:
        module = applier(module, granularity, wrap)

        if not isinstance(module, Fsdp2Module):
            module = wrap(module, reshard_after_forward=False)
    except (RuntimeError, ValueError) as ex:
        raise DistributedSetupError(
            "FSDP2 cannot be initialized. See the nested exception for details."
        ) from ex

    return module


def fsdp2_local_state_dict(module: Fsdp2Module) -> dict[str, object]:
    state_dict = {}

    sharded_state_dict = module.state_dict()

    fsdp_state = fully_shard.state(module)

    param_group = fsdp_state._fsdp_param_group
    if param_group is None:
        raise InvalidOperationError(
            "`module` does not have an associated FSDP2 parameter group."
        )

    device_mesh = param_group.mesh_info.mesh

    sdp_mesh_dim_idx = 0 if device_mesh.ndim == 1 else 1

    sdp_rank = device_mesh.get_local_rank(sdp_mesh_dim_idx)  # sharded

    for name, item in sharded_state_dict.items():
        if isinstance(item, DTensor):
            state_dict[name] = item.detach().to_local()
        # Save replicated items only on the first intra-node (i.e. sharded)
        # gang.
        elif sdp_rank == 0:
            if isinstance(item, Tensor):
                item = item.detach()

            state_dict[name] = item

    return state_dict


@contextmanager
def fsdp2_no_sync(module: Fsdp2Module) -> Iterator[None]:
    module.set_requires_gradient_sync(False)

    try:
        yield
    finally:
        module.set_requires_gradient_sync(True)


@contextmanager
def fsdp2_summon_full_parameters(module: Fsdp2Module) -> Iterator[None]:
    state = fully_shard.state(module)

    try:
        mp = state._mp_policy
    except AttributeError:
        mp = None

    def disable_hooks(module: Module, hook_name: str) -> None:
        backup_key = f"__fs2_{hook_name}_backup__"

        original_hooks = getattr(module, hook_name)

        hooks = OrderedDict()

        # Remove any FSDP2 hook.
        for handle, hook in original_hooks.items():
            try:
                hook_module = hook.__module__
            except AttributeError:
                hook_module = ""

            if hook_module.startswith("torch.distributed.fsdp"):
                continue

            hooks[handle] = hook

        setattr(module, backup_key, original_hooks)

        setattr(module, hook_name, hooks)

    def enable_hooks(module: Module, hook_name: str) -> None:
        backup_key = f"__fs2_{hook_name}_backup__"

        hooks = getattr(module, backup_key)

        setattr(module, hook_name, hooks)

        delattr(module, backup_key)

    def unshard(module: Module) -> None:
        for child in module.children():
            unshard(child)

        if isinstance(module, Fsdp2Module):
            module.unshard()

            disable_hooks(module, "_forward_hooks")
            disable_hooks(module, "_forward_pre_hooks")

    def reshard(module: Module) -> None:
        for child in module.children():
            reshard(child)

        if isinstance(module, Fsdp2Module):
            enable_hooks(module, "_forward_hooks")
            enable_hooks(module, "_forward_pre_hooks")

            module.reshard()

    def maybe_cast_dtype(t: Tensor) -> Tensor:
        if not isinstance(t, Parameter):
            return t

        if mp.param_dtype is None:
            return t

        return t.to(mp.param_dtype)

    unshard(module)

    apply_to_parameters(module, maybe_cast_dtype)

    try:
        yield
    finally:
        reshard(module)
