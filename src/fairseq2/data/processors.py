# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, List, Optional, Sequence

from torch import Tensor

from fairseq2 import _DOC_MODE
from fairseq2.data.typing import StringLike
from fairseq2.typing import DataType

if TYPE_CHECKING or _DOC_MODE:

    class StrSplitter:
        def __init__(self, sep: str = "\t") -> None:
            ...

        def __call__(self, s: StringLike) -> List[StringLike]:
            ...

    class StrToTensorConverter:
        def __init__(
            self,
            size: Optional[Sequence[int]] = None,
            dtype: Optional[DataType] = None,
        ) -> None:
            ...

        def __call__(self, s: StringLike) -> Tensor:
            ...

else:
    from fairseq2.C.data.processors import StrSplitter as StrSplitter
    from fairseq2.C.data.processors import StrToTensorConverter as StrToTensorConverter

    def _set_module_name() -> None:
        for t in [StrSplitter, StrToTensorConverter]:
            t.__module__ = __name__

    _set_module_name()
