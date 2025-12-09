# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.nn.data_parallel import (
    DataParallelFacade,
    _NoopDataParallelFacade,
    get_data_parallel_facade,
    set_data_parallel_facade,
)
from fairseq2.typing import ContextManager


class FooFacade(DataParallelFacade):
    def __init__(self, module: Module) -> None:
        self._module = module

    @override
    def state_dict(self) -> dict[str, object]:
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        pass

    @override
    def no_sync(self) -> ContextManager[None]:
        return nullcontext()

    @override
    def clip_grad_norm(self, max_norm: float | None) -> Tensor:
        return torch.empty()

    @override
    def summon_full_parameters(self) -> ContextManager[None]:
        return nullcontext()


def test_set_data_parallel_facade_works() -> None:
    m = Module()

    expected_facade = FooFacade(m)

    set_data_parallel_facade(m, expected_facade)

    facade = get_data_parallel_facade(m)

    assert facade is expected_facade


def test_get_data_parallel_facade_works_when_non_dp() -> None:
    m = Module()

    facade = get_data_parallel_facade(m)

    assert isinstance(facade, _NoopDataParallelFacade)
