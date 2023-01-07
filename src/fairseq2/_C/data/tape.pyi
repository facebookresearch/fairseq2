# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final

# fmt: off

@final
class Tape:
    def __init__(self) -> None:
        ...

    def rewind(self) -> None:
        ...
