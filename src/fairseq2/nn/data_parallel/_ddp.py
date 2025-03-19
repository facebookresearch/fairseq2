# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TypeAlias

import torch.distributed as dist
from torch import Tensor
from torch.distributed import GradBucket
from torch.futures import Future
from torch.nn import Module, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP

from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang, Gangs
from fairseq2.nn.data_parallel._error import DistributedSetupError
from fairseq2.nn.utils.module import (
    infer_device,
    reset_non_persistent_buffers,
    to_device,
    to_empty,
)

DdpModule: TypeAlias = DDP


def to_ddp(
    module: Module,
    gangs: Gangs,
    *,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
    normalize_gradients: bool = True,
) -> DdpModule:
    """Wrap ``module`` with DDP.

    :param module: The module to be wrapped with DDP.
    :param gangs: The gangs over which to replicate the module.
    :param find_unused_parameters: See the corresponding DDP documentation.
    :param static_graph: See the corresponding DDP documentation.
    :param normalize_gradients: If ``True``, normalizes gradients by the world
        size of the underlying process group.
    """
    if gangs.sdp.size > 1:
        raise NotSupportedError(
            "DDP does not support sharded data parallelism. Please use FSDP or FSDP2 instead."
        )

    try:
        module_device = infer_device(module)
    except ValueError as ex:
        raise DistributedSetupError(
            "The device of `module` is not valid. See the nested exception for details."
        ) from ex

    dp_gang = gangs.rdp  # replicated

    # DDP has no explicit support for meta initialization.
    if module_device.type == "meta":
        if dp_gang.rank == 0:
            # Since DDP always broadcasts the model from the coordinator process
            # materialize the model there.
            to_device(module, dp_gang.device)
        else:
            # For all other ranks, skip initialization as the parameters will be
            # overwritten anyways.
            to_empty(module, dp_gang.device)

            # Non-persistent buffers are never part of a module's state, so we
            # have to initialize them.
            reset_non_persistent_buffers(module)

    # Ensure that persistent buffers are in sync across all processes at
    # initialization.
    _broadcast_buffers(module, dp_gang)

    try:
        process_group = dp_gang.as_process_group()
    except NotSupportedError:
        raise DistributedSetupError(
            "The specified data parallel gang does not support conversion to a process group."
        ) from None

    try:
        module = DdpModule(
            module,
            broadcast_buffers=False,
            process_group=process_group,
            find_unused_parameters=find_unused_parameters,
            static_graph=static_graph,
        )
    except (RuntimeError, ValueError) as ex:
        raise DistributedSetupError(
            "DDP cannot be initialized. See the nested exception for details."
        ) from ex

    # For performance reasons we do not broadcast buffers during training, so
    # ensure that batch normalization works.
    SyncBatchNorm.convert_sync_batchnorm(module, process_group)

    # DDP, by default, normalizes gradients by the world size of the underlying
    # process group. For sequence-based tasks this is typically not ideal since
    # batch sizes can vary. Here, we disable that behavior if requested.
    if not normalize_gradients:
        module.register_comm_hook(state=dp_gang, hook=_allreduce_hook)

    return module


def _broadcast_buffers(module: Module, gang: Gang) -> None:
    memo: set[Tensor] = set()

    buffers = []

    state_dict = module.state_dict()

    for name, buffer in module.named_buffers():
        if buffer in memo:
            continue

        # Make sure that we don't broadcast non-persistent buffers since they
        # are static.
        if name in state_dict:
            buffers.append(buffer)

        memo.add(buffer)

    if not buffers:
        return

    pg = gang.as_process_group()

    bucket_size = 250 * 1024 * 1024  # Same as DDP bucket size.

    source_rank = 0

    from torch.distributed import _broadcast_coalesced

    # TODO(balioglu): Call c10d in fairseq2n instead.
    _broadcast_coalesced(pg, buffers, bucket_size, source_rank)


def _allreduce_hook(gang: Gang, bucket: GradBucket) -> Future[Tensor]:
    pg = gang.as_process_group()

    ft = dist.all_reduce(bucket.buffer(), group=pg, async_op=True).get_future()

    def return_reduced_bucket(f: Future[list[Tensor]]) -> Tensor:
        output = f.value()

        # Skip division by the world size.
        return output[0]

    return ft.then(return_reduced_bucket)  # type: ignore[no-any-return]
