# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import TYPE_CHECKING, Optional

from fairseq2n import DOC_MODE

from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.data.typing import PathLike, StringLike

if TYPE_CHECKING or DOC_MODE:

    class LineEnding(Enum):
        INFER = 0
        LF = 1
        CRLF = 2

    def read_text(
        pathname: PathLike,
        encoding: Optional[StringLike] = None,
        line_ending: LineEnding = LineEnding.INFER,
        ltrim: bool = False,
        rtrim: bool = False,
        skip_empty: bool = False,
        memory_map: bool = False,
        block_size: Optional[int] = None,
    ) -> DataPipelineBuilder:
        """Open a text file and return a data pipeline reading lines one by one."""
        ...

else:
    from fairseq2n.bindings.data.text.text_reader import LineEnding as LineEnding
    from fairseq2n.bindings.data.text.text_reader import read_text as read_text

    def _set_module_name() -> None:
        for t in [LineEnding, read_text]:
            t.__module__ = __name__

    _set_module_name()
