# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Protocol, final, runtime_checkable

from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDPModule
from typing_extensions import override

from fairseq2.checkpoint import CheckpointManager
from fairseq2.data_type import DataType
from fairseq2.error import InternalError
from fairseq2.gang import GangError, Gangs, raise_operational_gang_error
from fairseq2.logging import log
from fairseq2.nn.fsdp import FSDPApplier, FSDPWrapper
from fairseq2.nn.utils.module import to_device
from fairseq2.recipe.config import TrainerSection
from fairseq2.recipe.error import FSDPNotSupportedError
from fairseq2.recipe.internal.model import _ModelHolder
from fairseq2.runtime.lookup import Lookup


class _DataParallelModelWrapper(ABC):
    @abstractmethod
    def wrap(self, model_holder: _ModelHolder) -> Module: ...


@final
class _DelegatingDPModelWrapper(_DataParallelModelWrapper):
    def __init__(
        self,
        section: TrainerSection,
        gangs: Gangs,
        dp_wrappers: Lookup[_DataParallelModelWrapper],
    ) -> None:
        self._section = section
        self._gangs = gangs
        self._dp_wrappers = dp_wrappers

    @override
    def wrap(self, model_holder: _ModelHolder) -> Module:
        if self._gangs.dp.size == 1:
            to_device(model_holder.model, self._gangs.root.device)

            return model_holder.model

        data_parallelism = self._section.data_parallelism

        if data_parallelism == "fsdp":
            if self._gangs.rdp.size > 1 and self._gangs.sdp.size == 1:
                log.warning("Hybrid sharded data parallelism not enabled. Falling back to DDP.")  # fmt: skip

                data_parallelism = "ddp"

        wrapper = self._dp_wrappers.maybe_get(data_parallelism)
        if wrapper is None:
            raise InternalError(f"`section.data_parallelism` is '{data_parallelism}'.")

        return wrapper.wrap(model_holder)


@runtime_checkable
class _DDPFactory(Protocol):
    def __call__(
        self, module: Module, gangs: Gangs, *, find_unused_parameters: bool
    ) -> DDPModule: ...


@final
class _DDPModelWrapper(_DataParallelModelWrapper):
    def __init__(
        self, ddp_factory: _DDPFactory, gangs: Gangs, static_graph: bool
    ) -> None:
        self._ddp_factory = ddp_factory
        self._gangs = gangs
        self._static_graph = static_graph

    @override
    def wrap(self, model_holder: _ModelHolder) -> Module:
        log.info("Wrapping model with DDP and broadcasting to all processes.")

        # We do not set DDP's `static_graph` parameter. Unfortunately, support for
        # that feature is finicky in DDP. `find_unused_parameters` is still useful
        # though and can have measurable impact on performance.
        try:
            dp_model = self._ddp_factory(
                model_holder.model,
                self._gangs,
                find_unused_parameters=not self._static_graph,
            )
        except GangError as ex:
            raise_operational_gang_error(ex)

        log.info("Model wrapped with DDP and broadcasted.")

        return dp_model


@runtime_checkable
class _FSDPFactory(Protocol):
    def __call__(
        self,
        module: Module,
        gangs: Gangs,
        applier: FSDPApplier,
        *,
        version: Literal["v1", "v2"],
        mixed_precision_dtype: DataType | None,
        fp32_reduce: bool,
        broadcast_state: bool,
        skip_init: bool,
        reshard_after_forward: bool,
    ) -> Module: ...


@final
class _FSDPModelWrapper(_DataParallelModelWrapper):
    def __init__(
        self,
        fsdp_factory: _FSDPFactory,
        section: TrainerSection,
        checkpoint_manager: CheckpointManager,
        gangs: Gangs,
    ) -> None:
        self._fsdp_factory = fsdp_factory
        self._section = section
        self._checkpoint_manager = checkpoint_manager
        self._gangs = gangs

    @override
    def wrap(self, model_holder: _ModelHolder) -> Module:
        if not model_holder.family.supports_fsdp:
            raise FSDPNotSupportedError()

        fsdp_config = self._section.fsdp

        def apply_fsdp(module: Module, wrapper: FSDPWrapper) -> Module:
            if fsdp_config.granularity == "model":
                return wrapper(module, reshard_after_forward=False)

            return model_holder.family.apply_fsdp(
                module, fsdp_config.granularity, wrapper
            )

        if self._section.mixed_precision.mode == "static":
            mp_dtype = self._section.mixed_precision.dtype
        else:
            mp_dtype = None

        has_checkpoint = self._checkpoint_manager.has_checkpoint()

        if self._section.fsdp.version == "v1":
            if has_checkpoint:
                log.info("Wrapping model with FSDP1.")  # fmt: skip
            else:
                log.info("Wrapping model with FSDP1 and broadcasting to all processes.")  # fmt: skip

            try:
                dp_model = self._fsdp_factory(
                    model_holder.model,
                    self._gangs,
                    apply_fsdp,
                    version="v1",
                    mixed_precision_dtype=mp_dtype,
                    fp32_reduce=fsdp_config.fp32_reduce,
                    broadcast_state=not has_checkpoint,
                    skip_init=has_checkpoint,
                    reshard_after_forward=fsdp_config.reshard_after_forward,
                )
            except GangError as ex:
                raise_operational_gang_error(ex)

            if has_checkpoint:
                log.info("Model wrapped with FSDP1.")
            else:
                log.info("Model wrapped with FSDP1 and broadcasted.")
        else:
            if has_checkpoint:
                log.info("Wrapping model with FSDP2.")  # fmt: skip
            else:
                log.info("Wrapping model with FSDP2 and broadcasting to all processes.")  # fmt: skip

            try:
                dp_model = self._fsdp_factory(
                    model_holder.model,
                    self._gangs,
                    apply_fsdp,
                    version="v2",
                    mixed_precision_dtype=mp_dtype,
                    fp32_reduce=fsdp_config.fp32_reduce,
                    broadcast_state=not has_checkpoint,
                    skip_init=has_checkpoint,
                    reshard_after_forward=fsdp_config.reshard_after_forward,
                )
            except GangError as ex:
                raise_operational_gang_error(ex)

            if has_checkpoint:
                log.info("Model wrapped with FSDP2.")
            else:
                log.info("Model wrapped with FSDP2 and broadcasted.")

        return dp_model
