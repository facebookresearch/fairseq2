# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

from fairseq2.device import determine_default_device
from fairseq2.gang import Gang, setup_default_gang, setup_parallel_gangs
from fairseq2.logging import LogWriter
from fairseq2.models.fsdp import get_fsdp_wrap_policy
from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.nn.utils.module import broadcast_module, to_device
from fairseq2.recipes.utils.log import log_environment_info


def setup_root_gang(
    log: LogWriter,
    *,
    timeout: timedelta | None = None,
    monitored: bool = False,
) -> Gang:
    """Set up the root gang.

    :param log:
        The log to write to.
    :param timeout:
        The timeout for collective operations.
    :param monitored:
        If ``True``,  puts a monitored barrier before every collective call.
    """
    device = determine_default_device()

    log_environment_info(log, device)

    # In case we run on Ampere or later, use TF32.
    torch.set_float32_matmul_precision("high")

    log.info("Initializing the root gang.")

    gang = setup_default_gang(timeout=timeout, monitored=monitored)

    log.info("Root gang initialized.")

    return gang


def setup_gangs(
    log: LogWriter,
    *,
    tp_size: int = 1,
    timeout: timedelta | None = None,
    monitored: bool = False,
) -> tuple[Gang, dict[str, Gang]]:
    """Set up the root, data, and tensor parallel gangs.

    :param log:
        The log to write to.
    :param tp_size:
        The size of tensor parallel gangs.
    :param timeout:
        The timeout for collective operations.
    :param monitored:
        If ``True``,  puts a monitored barrier before every collective call.
    """
    root_gang = setup_root_gang(log, timeout=timeout, monitored=monitored)

    log.info("Initializing data and tensor parallel gangs.")

    try:
        gangs = setup_parallel_gangs(root_gang, tp_size=tp_size)
    except ValueError as ex:
        raise RuntimeError(
            f"The size of the root gang ({root_gang.size}) is not divisible by `tensor_parallel_size` ({tp_size})."
        ) from ex

    log.info("Data and tensor parallel gangs initialized.")

    return root_gang, {"dp": gangs.dp, "tp": gangs.tp}


def broadcast_model(model: Module, gang: Gang, log: LogWriter) -> None:
    """Broadcast ``model`` to all processes in ``gang``."""
    log.info("Broadcasting the model to all processes.")

    broadcast_module(model, gang)

    log.info("Model broadcasted.")


def to_data_parallel(
    model: Module,
    gang: Gang,
    parallelism: Literal["ddp", "fsdp"],
    log: LogWriter,
    **kwargs: Any,
) -> Module:
    """Wrap ``model`` with DDP or FSDP.

    :param model:
        The model to wrap.
    :param gang:
        The gang over which to distribute data.
    :param parallelism:
        The parallelism API to use.
    :param log:
        The log to write to.
    :param kwargs:
        The keyword arguments to pass to :func:`to_ddp` or :func:`to_fsdp`. The
        parameter names should be prefixed with 'ddp_' and 'fsdp_' respectively.
    """

    if parallelism == "ddp":
        if gang.size == 1:
            to_device(model, gang.device)

            return model

        ddp_args = {}

        for key, value in kwargs.items():
            if key.startswith("ddp_"):
                ddp_args[key[4:]] = value

        log.info("Wrapping the model with DDP and broadcasting to all processes.")

        model = to_ddp(model, gang, **ddp_args)

        log.info("Model wrapped with DDP and broadcasted.")

        return model

    if parallelism == "fsdp":
        if gang.size == 1:
            to_device(model, gang.device)

            return model

        fsdp_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("fsdp_"):
                fsdp_kwargs[key[5:]] = value

        broadcast_state = fsdp_kwargs.get("broadcast_state", False)

        if not broadcast_state:
            log.info("Wrapping the model with FSDP.")
        else:
            log.info("Wrapping the model with FSDP and broadcasting to all processes.")

        wrap_policy, ignored_modules = get_fsdp_wrap_policy(
            model, wrap_granularity=fsdp_kwargs.pop("wrap_granularity", "stack")
        )

        model = to_fsdp(
            model,
            gang,
            wrap_policy,
            ignored_modules=ignored_modules,
            **fsdp_kwargs,
        )

        if not broadcast_state:
            log.info("Model wrapped with FSDP.")
        else:
            log.info("Model wrapped with FSDP and broadcasted.")

        return model

    raise ValueError(
        f"`data_parallelism` must be 'ddp' or 'fsdp', but is '{parallelism}' instead."
    )


def compile_model(model: Module, log: LogWriter, *, dynamic: bool = True) -> Module:
    """Apply :func:`torch.compile` to ``model``."""
    log.info("Applying `torch.compile()` to the model.")

    return torch.compile(  # type: ignore[return-value]
        model, dynamic=dynamic, options={"shape_padding": dynamic}
    )


def check_model_type(model: Module, kls: type[Module]) -> None:
    """Check if a potentially DDP or FSDP wrapped `model` is of type `kls`."""
    if isinstance(model, (DDP, FSDP)):
        model = model.module

    if not isinstance(model, kls):
        raise ValueError(
            f"`model` must be of type `{kls}`, but is of type `{type(model)}` instead."
        )
