# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Optional, Tuple

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import Module

from fairseq2.gang import Gang
from fairseq2.nn.fsdp import FSDPWrapPolicy
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder


def get_transformer_wrap_policy(
    model: Module, gang: Gang
) -> Tuple[Optional[FSDPWrapPolicy], Optional[List[str]]]:
    """See :class:`~fairseq2.models.fsdp.FSDPWrapPolicyProvider`."""
    kls = (TransformerEncoder, TransformerDecoder)

    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=kls)

    return wrap_policy, None
