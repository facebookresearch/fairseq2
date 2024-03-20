# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

from fairseq2.gang import Gang


def to_ddp(
    module: Module,
    gang: Gang,
    *,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> DDP:
    """Wrap ``module`` with DDP.

    :param module:
        The module to be wrapped with DDP.
    :param gang:
        The gang over which the module will be replicated.
    :param find_unused_parameters:
        See the corresponding DDP documentation.
    :param static_graph:
        See the corresponding DDP documentation.
    """
    return DDP(
        module,
        device_ids=[gang.device],
        process_group=gang.as_process_group(),
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
    )
