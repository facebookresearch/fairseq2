# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from collections.abc import Mapping

import torch
from torch import Tensor

from fairseq2.sharder import ShardSpec


def reshard_tensor(
    key: str,
    source_splits: list[list[Tensor]],
    source_shard_sizes: tuple[int, int],
    target_shard_sizes: tuple[int, int],
    target_shard_ranks: tuple[int, int],
    shard_specs: Mapping[str, ShardSpec] | None,
) -> Tensor:
    source_tp_size, source_dp_size = source_shard_sizes
    target_tp_size, target_dp_size = target_shard_sizes

    target_tp_rank, target_dp_rank = target_shard_ranks

    # If the source and target tensor parallel sizes match, we can directly
    # return the unsharded data parallel tensor.
    if source_tp_size == target_tp_size:
        source_dp_splits = source_splits[target_tp_rank]

        if source_dp_size == 1:
            return source_dp_splits[0]

        return torch.cat(source_dp_splits, dim=0)

    tp_dim = _get_tp_dim(key, shard_specs)

    # We assume that non-tensor parallel parameters are always replicated.
    if tp_dim == -1:
        source_dp_splits = source_splits[0]

        if source_dp_size == 1:
            return source_dp_splits[0]

        return torch.cat(source_dp_splits, dim=0)

    tp_splits = []

    # Unshard the tensor over the source tensor parallel dimension.
    for source_dp_splits in source_splits:
        if source_dp_size == 1:
            tp_split = source_dp_splits[0]
        else:
            tp_split = torch.cat(source_dp_splits, dim=0)

        tp_splits.append(tp_split)

    # Reshard the tensor over the target parallel dimension.
    source_tp_dim_size = tp_splits[0].size(tp_dim)

    tp_dim_size = source_tp_dim_size * source_tp_size

    target_tp_dim_size = tp_dim_size // target_tp_size

    f_target_idx = target_tp_rank * target_tp_dim_size
    l_target_idx = target_tp_rank * target_tp_dim_size + target_tp_dim_size - 1

    f_source_tp_shard_idx = f_target_idx // source_tp_dim_size
    l_source_tp_shard_idx = l_target_idx // source_tp_dim_size

    f_source_idx = f_source_tp_shard_idx * source_tp_dim_size

    tp_sub_splits = []

    for idx in range(f_source_tp_shard_idx, l_source_tp_shard_idx + 1):
        tp_sub_splits.append(tp_splits[idx])

    del tp_splits

    tensor = torch.cat(tp_sub_splits, dim=tp_dim)

    del tp_sub_splits

    return tensor.narrow(
        dim=tp_dim, start=f_target_idx - f_source_idx, length=target_tp_dim_size
    )


def _get_tp_dim(key: str, shard_specs: Mapping[str, ShardSpec] | None) -> int:
    if shard_specs is None:
        return -1

    offset = key.rfind(".")
    if offset >= 0:
        module_name = key[:offset]
    else:
        module_name = key

    for pattern, spec in shard_specs.items():
        if re.match(pattern, module_name):
            return spec.dim

    return -1
