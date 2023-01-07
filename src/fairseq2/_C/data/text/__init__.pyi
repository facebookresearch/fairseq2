# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.data.typing import StringLike

# fmt: off

class LineEnding(Enum):
    INFER = 0
    LF = 1
    CRLF = 2


def read_text(
    pathname: StringLike,
    encoding: StringLike = "",
    line_ending: LineEnding = LineEnding.INFER,
    ltrim: bool = False,
    rtrim: bool = False,
    skip_empty: bool = False,
    memory_map: bool = False,
    block_size: int | None = None,
) -> DataPipelineBuilder:
    ...
