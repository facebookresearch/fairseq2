# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from fairseq2n import DOC_MODE

from fairseq2.data.data_pipeline import DataPipelineBuilder

if TYPE_CHECKING or DOC_MODE:

    class LineEnding(Enum):
        INFER = 0
        LF = 1
        CRLF = 2

    def read_text(
        path: Path,
        key: str | None = None,
        encoding: str | None = None,
        line_ending: LineEnding = LineEnding.INFER,
        ltrim: bool = False,
        rtrim: bool = False,
        skip_empty: bool = False,
        memory_map: bool = False,
        block_size: int | None = None,
    ) -> DataPipelineBuilder:
        """Open a text file and return a data pipeline reading lines one by one."""
        ...

else:
    from fairseq2n.bindings.data.text.text_reader import (  # noqa: F401
        LineEnding as LineEnding,
    )
    from fairseq2n.bindings.data.text.text_reader import (  # noqa: F401
        read_text as read_text,
    )
