# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

from torch.nn import Module

from fairseq2.gang import Gangs


def compile_model(model: Module, gangs: Gangs) -> Module:
    return cast(Module, model.compile())
