# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext

from torch.distributed._shard import load_with_process_group

from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.typing import ContextManager


def load_with_sdp_gang(gangs: Gangs) -> ContextManager:
    try:
        pg = gangs.sdp.as_process_group()
    except NotSupportedError:
        return nullcontext()

    return load_with_process_group(pg)
