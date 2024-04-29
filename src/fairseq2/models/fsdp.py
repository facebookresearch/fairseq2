# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Literal, Optional, Set, Tuple, Type

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import Module

from fairseq2.nn.fsdp import FSDPWrapPolicy
from fairseq2.nn.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


def get_fsdp_wrap_policy(
    model: Module, wrap_granularity: Literal["layer", "stack", "model"] = "layer"
) -> Tuple[Optional[FSDPWrapPolicy], Optional[List[Module]]]:
    """Return the FSDP wrap policy for ``model`` along with ignored modules.

    :param model:
        The model to be wrapped.
    :param wrap_granularity:
        The granularity at which to wrap modules of ``model``.

          - 'layer': Wraps individual layers (e.g. :class:`TransformerDecoderLayer`).
          - 'stack': Wraps layer stacks (e.g. :class:`TransformerDecoder`).
          - 'model': Wraps ``model`` only.
    """
    if wrap_granularity == "model":
        return None, None

    kls: Set[Type[Module]]

    if wrap_granularity == "stack":
        kls = {TransformerEncoder, TransformerDecoder}
    elif wrap_granularity == "layer":
        kls = {TransformerEncoderLayer, TransformerDecoderLayer}
    else:
        raise ValueError(
            f"`wrap_granularity` must be 'layer', 'stack', or 'model', but is '{wrap_granularity}' instead."
        )

    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=kls)

    return wrap_policy, None
