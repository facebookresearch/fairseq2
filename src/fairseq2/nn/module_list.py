# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Optional, final

from torch import Generator
from torch.nn import Module
from torch.nn import ModuleList as TorchModuleList


# compat
@final
class ModuleList(TorchModuleList):
    def __init__(
        self,
        modules: Optional[Iterable[Module]] = None,
        *,
        drop_p: float = 0.0,
        generator: Optional[Generator] = None,
    ) -> None:
        super().__init__(modules)

    def drop_iter(self) -> Any:
        return super().__iter__()
