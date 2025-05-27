# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn, final

from torch.nn import Module

from fairseq2.data_type import DataType
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.typing import ContextManager

# isort: split

from fairseq2.nn.data_parallel._common import FSDPApplier, FSDPGranularity


@final
class FSDP2(Module):
    def unshard(self) -> None:
        _raise_error()

    def reshard(self) -> None:
        _raise_error()

    def set_requires_grad_sync(self, value: bool) -> None:
        _raise_error()


def to_fsdp2(
    module: Module,
    gangs: Gangs,
    applier: FSDPApplier,
    *,
    granularity: FSDPGranularity = "layer",
    mixed_precision_dtype: DataType | None = None,
    fp32_reduce: bool = False,
    broadcast_state: bool = False,
    skip_init: bool = False,
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
) -> FSDP2:
    _raise_error()


def fsdp2_local_state_dict(fsdp_module: FSDP2) -> dict[str, object]:
    _raise_error()


def fsdp2_load_local_state_dict(
    module: FSDP2, state_dict: Mapping[str, object]
) -> None:
    _raise_error()


def fsdp2_no_sync(fsdp_module: FSDP2) -> ContextManager:
    _raise_error()


def fsdp2_summon_full_parameters(fsdp_module: FSDP2) -> ContextManager:
    _raise_error()


def _raise_error() -> NoReturn:
    raise NotSupportedError("FSDP2 is only supported by PyTorch 2.6 and later.")
