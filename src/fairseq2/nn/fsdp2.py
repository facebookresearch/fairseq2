# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional

import itertools
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.distributed._tensor import (
    distribute_tensor
)
from torch.nn import Module
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)


from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.nn.transformer import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,    
)
from fairseq2.nn.fsdp import FSDPWrapPolicy, FSDPMemoryPolicy, FSDP_STANDARD_MEMORY_POLICY
from fairseq2.typing import DataType

log = get_log_writer(__name__)


def to_fsdp2(
    module: Module,
    gang: Gang,
    wrap_policy: Optional[FSDPWrapPolicy],
    *,
    skip_init: bool = False,
    memory_policy: Optional[FSDPMemoryPolicy] = None,
    reshard_after_forward: bool = True,
    local_world_size: Optional[int] = None,
    mixed_precision_dtype: Optional[DataType] = None,
    fp32_reduce: bool = False,
    activation_checkpointing: bool = False,    
    **kwargs: Any,
) -> Module:
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

    if memory_policy is None:
        memory_policy = FSDP_STANDARD_MEMORY_POLICY

    if mixed_precision_dtype is None:
        mp_policy = None
    elif mixed_precision_dtype == torch.float32:
        mp_policy = None
    else:
        reduce_dtype = torch.float32 if fp32_reduce else mixed_precision_dtype

        mp_policy = MixedPrecisionPolicy(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=reduce_dtype,
        )

    fsdp_kwargs: Dict[str, Any] = {"mp_policy": mp_policy}

    if memory_policy.cpu_offload:
        from torch.distributed._composable.fsdp import CPUOffloadPolicy

        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if gang.rank == 0:
        full_sd = module.state_dict()    

    for m in module.modules():
        # TransformerEncoder/DecoderLayer is wrapped by CheckpointWrapper
        # when activation_checkpointing
        if activation_checkpointing:
            if isinstance(m, CheckpointWrapper):
                fully_shard(m, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)
        else:
            if isinstance(m, TransformerEncoderLayer) or isinstance(m, TransformerDecoderLayer):
                fully_shard(m, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)

    fully_shard(module, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)
    
    # FSDP2 allows materializing tensors onto GPU after sharding
    # Construct a sharded state dict from the rank 0 full state dict by
    # broadcasting and sharding
    meta_sharded_sd = module.state_dict()
    sharded_sd = {}
    if gang.rank == 0:
        for (param_name, full_param), sharded_meta_param in zip(
            full_sd.items(), meta_sharded_sd.values()
        ):
            full_param = full_param.detach().cuda()
            mesh = sharded_meta_param.device_mesh
            dist.broadcast(full_param, src=0, group=mesh.get_group(0))
            sharded_tensor = distribute_tensor(
                full_param, mesh, sharded_meta_param.placements
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    else:
        for param_name, sharded_meta_param in meta_sharded_sd.items():
            full_tensor = torch.empty(
                sharded_meta_param.size(),
                device="cuda",
                dtype=sharded_meta_param.dtype,
            )
            mesh = sharded_meta_param.device_mesh
            dist.broadcast(full_tensor, src=0, group=mesh.get_group(0))
            sharded_tensor = distribute_tensor(
                full_tensor, mesh, sharded_meta_param.placements
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)

    module.load_state_dict(sharded_sd, assign=True)

    for buffer_name, buffer in module.named_buffers():
        if buffer_name not in sharded_sd:
            if gang.rank != 0:
                full_buffer = torch.empty(
                    buffer.size(),
                    device="cuda",
                    dtype=buffer.dtype,
                )
                mesh = sharded_meta_param.device_mesh
                dist.broadcast(full_buffer, src=0, group=mesh.get_group(0))

                splits = buffer_name.split('.')
                key = '.'.join(splits[:-1])
                submodule = module.get_submodule(key)
                setattr(submodule, splits[-1], full_buffer)
            else:
                mesh = sharded_meta_param.device_mesh
                dist.broadcast(buffer, src=0, group=mesh.get_group(0))

    for name, tensor in itertools.chain(module.named_parameters(), module.named_buffers()):
        assert tensor.device != torch.device("meta"), name

    return module