# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def reshard_tensor(
    key: str,
    source_splits: list[list[Tensor]],
    source_shard_sizes: tuple[int, int],
    target_shard_sizes: tuple[int, int],
    target_shard_ranks: tuple[int, int],
    shard_dims: Mapping[str, int],
) -> Tensor:
    """
    Reshards a parameter tensor from a distributed source configuration to a
    target configuration.

    This function is meant for authors of new :class:`ModelCheckpointLoader`
    implementations. It handles the complex task of resharding tensors when
    loading checkpoints from one distributed configuration (e.g. 4-way tensor
    parallelism) to a different target configuration (e.g. 8-way tensor
    parallelism). It efficiently concatenates and slices tensors to produce the
    correct shards for the target rank.

    The resharding process involves:

    1. Determining if the tensor requires tensor parallelism based on specified
       shard dimensions.
    2. For tensor parallel tensors, concatenating source shards and re-slicing
       for the target configuration in a memory-efficient way.
    3. For replicated tensors, concatenating data parallel splits.

    ``key`` specifies the name of the parameter to retrieve its sharding
    information from ``shard_dims``. See :func:`~fairseq2.nn.get_sharding_dims`
    for more information.

    ``source_splits`` is a 2D list structure ``[tp_idx][dp_idx]`` containing the
    source tensor shards. The outer list specifies tensor parallel shards and
    inner lists specify data parallel shards.

    ``source_shard_sizes`` and ``target_shard_sizes`` specify the distributed
    source and target configurations respectively in the form of ``(tp_size, dp_size)``.

    ``target_shard_ranks`` specifies the ranks of the current process in the
    target configuration in the form of ``(tp_rank, dp_rank)``.

    ``shard_dims`` specifies the mapping from parameter names to dimensions
    along which parameters should be sharded for tensor parallelism. Omitted for
    replicated tensors. See :func:`~fairseq2.nn.get_sharding_dims` for more
    information.

    Returns the resharded tensor for the target rank and configuration.

    .. code:: python
        :caption: Resharding from 2-way TP to 4-way TP

        param_name = "model.weight"

        # 2 TP shards with 1 DP shard each.
        source_splits = [[tensor_tp0_dp0], [tensor_tp1_dp0]]

        source_shard_sizes = (2, 1)  # 2-way TP, 1-way DP
        target_shard_sizes = (4, 1)  # 4-way TP, 1-way DP

        target_shard_ranks = (2, 0)  # Want shard for TP rank 2

        # For a tensor with TP dim=0, this will concatenate the 2 source shards
        # and slice out the portion corresponding to TP rank 2 in 4-way setup
        resharded = reshard_tensor(
            param_name,
            source_splits,
            source_shard_sizes,
            target_shard_sizes,
            target_shard_ranks,
            shard_dims={param_name: 0},
        )

    .. note::

        This function deletes intermediate tensors during the resharding process
        to minimize peak memory usage.
    """
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

    tp_dim = shard_dims.get(key, None)

    # We assume that non-tensor parallel parameters are always replicated.
    if tp_dim is None:
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

    # Source shard dimensions
    source_tp_dim_size = tp_splits[0].size(tp_dim)

    # Total unsharded dimension size
    tp_dim_size = source_tp_dim_size * source_tp_size

    # Target shard dimension
    target_tp_dim_size = tp_dim_size // target_tp_size

    # Slice boundaries for this target rank.
    first_target_idx = target_tp_rank * target_tp_dim_size
    last_target_idx = target_tp_rank * target_tp_dim_size + target_tp_dim_size - 1

    # Determine which source ranks contain the target slice.
    first_source_tp_shard_idx = first_target_idx // source_tp_dim_size
    last_source_tp_shard_idx = last_target_idx // source_tp_dim_size

    # Starting index of the first relevant source shard
    first_source_idx = first_source_tp_shard_idx * source_tp_dim_size

    # Collect sub-slices from relevant source shards.
    tp_sub_splits = []

    for idx in range(first_source_tp_shard_idx, last_source_tp_shard_idx + 1):
        tp_sub_splits.append(tp_splits[idx])

    del tp_splits

    tensor = torch.cat(tp_sub_splits, dim=tp_dim)

    del tp_sub_splits

    # Extract exact slice needed for this target rank.
    return tensor.narrow(
        dim=tp_dim, start=first_target_idx - first_source_idx, length=target_tp_dim_size
    )
