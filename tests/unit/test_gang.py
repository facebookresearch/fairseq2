# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from typing_extensions import override

from fairseq2.gang import (
    AbstractGang,
    GangError,
    ReduceOperation,
    _setup_2D_mesh_gangs,
    setup_hybrid_fsdp_gangs,
    setup_parallel_gangs,
)


class MockGang(AbstractGang):
    """
    A mock gang that keeps track of the list of the process ranks.
    """

    _group_ranks: list[int]

    def __init__(self, group_ranks: list[int], *, rank: int = 0) -> None:
        super().__init__(rank=rank, size=len(group_ranks), device=torch.device("cpu"))
        self._group_ranks = list(group_ranks)

    @override
    def close(self) -> None:
        pass

    @override
    def _do_make_gang(self, ranks: Sequence[int]) -> MockGang | None:
        try:
            idx = ranks.index(self._rank)
        except ValueError:
            return None

        return MockGang(list(ranks), rank=idx)

    @property
    def group_ranks(self) -> list[int]:
        return self._group_ranks

    @override
    def as_process_group(self) -> ProcessGroup:
        raise RuntimeError("This method should not be called for this mock gang.")

    @override
    def barrier(self) -> None:
        pass

    @override
    def all_reduce(self, tensor: Tensor, op: ReduceOperation) -> None:
        raise RuntimeError("This method should not be called for this mock gang.")

    @override
    def all_gather(self, output_tensor: Tensor, input_tensor: Tensor) -> None:
        raise RuntimeError("This method should not be called for this mock gang.")

    @override
    def all_gather_to_list(
        self, output_tensors: list[Tensor], input_tensor: Tensor
    ) -> None:
        raise RuntimeError("This method should not be called for this mock gang.")

    @override
    def broadcast(self, tensor: Tensor, source_rank: int = 0) -> None:
        raise RuntimeError("This method should not be called for this mock gang.")

    @override
    def broadcast_objects(self, objects: list[Any], source_rank: int = 0) -> None:
        raise RuntimeError("This method should not be called for this mock gang.")


class TestGang:
    @pytest.mark.parametrize(
        "rank,size,row_length,expected",
        [
            (0, 2, 2, ([0, 1], [0])),
            # mesh for 2 hosts, 2 GPUs each:
            # Host 0: g0 | g1
            # Host 1: g2 | g3
            (0, 4, 2, ([0, 1], [0, 2])),
            (1, 4, 2, ([0, 1], [1, 3])),
            (2, 4, 2, ([2, 3], [0, 2])),
            (0, 8, 4, ([0, 1, 2, 3], [0, 4])),
        ],
    )
    def test_setup_2D_mesh_works(
        self,
        rank: int,
        size: int,
        row_length: int,
        expected: tuple[list[int], list[int]],
    ) -> None:
        root_gang = MockGang(list(range(size)), rank=rank)

        gangs = _setup_2D_mesh_gangs(
            root_gang,
            row_length=row_length,
            create_single_rank_process_groups=True,
        )

        for i in range(2):
            gang = gangs[i]

            # typecheck confirms that `create_single_rank_process_groups` works
            assert isinstance(gang, MockGang)

            assert gang.group_ranks == expected[i]

    @pytest.mark.parametrize(
        "rank,size,row_length",
        [
            (0, 2, 0),
            (0, 2, 3),
            (0, 16, 7),
        ],
    )
    def test_setup_with_2D_mesh_raises_exception_on_bad_mesh(
        self, rank: int, size: int, row_length: int
    ) -> None:
        root_gang = MockGang(list(range(size)), rank=rank)

        with pytest.raises((ValueError, GangError)):
            setup_hybrid_fsdp_gangs(root_gang, row_length)

        with pytest.raises((ValueError, GangError)):
            setup_parallel_gangs(root_gang, tp_size=row_length)
