# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

from torch import Tensor

from fairseq2 import _DOC_MODE
from fairseq2.data.typing import StringLike
from fairseq2.typing import DataType

if TYPE_CHECKING or _DOC_MODE:

    class StrSplitter:
        def __init__(
            self,
            sep: str = "\t",
            names: Optional[Sequence[str]] = None,
            indices: Optional[Sequence[int]] = None,
            exclude: bool = False,
        ) -> None:
            ...

        def __call__(
            self, s: StringLike
        ) -> Union[List[StringLike], Dict[str, StringLike]]:
            ...

    class StrToIntConverter:
        def __init__(self, base: int = 10) -> None:
            ...

        def __call__(self, s: StringLike) -> int:
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
    from fairseq2n.bindings.data.text.converters import StrSplitter as StrSplitter
    from fairseq2n.bindings.data.text.converters import (
        StrToIntConverter as StrToIntConverter,
    )
    from fairseq2n.bindings.data.text.converters import (
        StrToTensorConverter as StrToTensorConverter,
    )

    def _set_module_name() -> None:
        for t in [StrSplitter, StrToIntConverter, StrToTensorConverter]:
            t.__module__ = __name__

    _set_module_name()
