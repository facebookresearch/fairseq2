# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Mapping

from torch.nn import Module

from fairseq2.gang import Gang
from fairseq2.logging import LogWriter
from fairseq2.models import load_model
from fairseq2.nn.utils.module import freeze_parameters
from fairseq2.recipes.utils.asset import AssetReference, retrieve_asset_card
from fairseq2.recipes.utils.setup import broadcast_model
from fairseq2.typing import META, DataType


def _load_reference_model(
    model_name_or_card: AssetReference,
    dtype: DataType,
    root_gang: Gang,
    gangs: Mapping[str, Gang],
    tensor_parallel_size: int,
    log: LogWriter,
) -> Module:
    dp_gang = gangs["dp"]

    card = retrieve_asset_card(model_name_or_card)

    log.info("Loading {} reference model on data parallel rank 0 (per shard).", card.name)  # fmt: skip

    if dp_gang.rank == 0:
        init_device = root_gang.device
    else:
        init_device = META

    # TODO: figure out how to load the reference model onto its own gangs
    model = load_model(card, gangs=gangs, device=init_device, dtype=dtype)

    root_gang.barrier()

    log.info("Reference model loaded on data parallel rank 0.")

    model.eval()

    freeze_parameters(model)

    # Distribute the model to all processes in the gang.
    if dp_gang.size != 1:
        broadcast_model(model, dp_gang, log)

    return model
