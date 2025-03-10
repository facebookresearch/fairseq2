# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import NoReturn, final

from torch.nn import Module

from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.nn.data_parallel._common import FsdpApplier, FsdpGranularity
from fairseq2.typing import ContextManager, DataType


@final
class Fsdp2Module(Module):
    def unshard(self) -> None:
        _raise_error()

    def reshard(self) -> None:
        _raise_error()

    def set_requires_gradient_sync(self, value: bool) -> None:
        _raise_error()


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
    _raise_error()


def fsdp2_local_state_dict(fsdp_module: Fsdp2Module) -> dict[str, object]:
    _raise_error()


def fsdp2_no_sync(fsdp_module: Fsdp2Module) -> ContextManager:
    _raise_error()


def fsdp2_summon_full_parameters(fsdp_module: Fsdp2Module) -> ContextManager:
    _raise_error()


def _raise_error() -> NoReturn:
    raise NotSupportedError("FSDP2 requires PyTorch 2.6 or later.")
