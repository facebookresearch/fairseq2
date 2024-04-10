# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch.distributed as dist
from torch import Tensor
from torch.distributed import GradBucket
from torch.futures import Future
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

from fairseq2.gang import Gang


def to_ddp(
    module: Module,
    gang: Gang,
    *,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
    normalize_gradients: bool = True,
) -> DDP:
    """Wrap ``module`` with DDP.

    :param module:
        The module to be wrapped with DDP.
    :param gang:
        The gang over which to replicate the module.
    :param find_unused_parameters:
        See the corresponding DDP documentation.
    :param static_graph:
        See the corresponding DDP documentation.
    :param normalize_gradients:
        If ``True``, normalizes gradients by the world size of the underlying
        process group.
    """
    ddp = DDP(
        module,
        device_ids=[gang.device],
        process_group=gang.as_process_group(),
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
    )

    # DDP, by default, normalizes gradients by the world size of the underlying
    # process group. For sequence-based tasks this is typically not ideal since
    # batch sizes can vary. Here, we disable that behavior.
    if not normalize_gradients:
        ddp.register_comm_hook(state=gang, hook=_allreduce_hook)

    return ddp


def _allreduce_hook(gang: Gang, bucket: GradBucket) -> Future[Tensor]:
    pg = gang.as_process_group()

    ft = dist.all_reduce(bucket.buffer(), group=pg, async_op=True).get_future()

    def return_reduced_bucket(f: Future[List[Tensor]]) -> Tensor:
        output = f.value()

        # Skip division by the world size.
        return output[0]

    return ft.then(return_reduced_bucket)  # type: ignore[no-any-return]
