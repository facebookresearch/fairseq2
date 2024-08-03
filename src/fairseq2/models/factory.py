# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy_extensions import DefaultNamedArg
from torch.nn import Module

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.typing import DataType, Device

if TYPE_CHECKING:  # compat: remove when Python 3.9 support is dropped.
    model_factories = ConfigBoundFactoryRegistry[
        [DefaultNamedArg(Device, "device"), DefaultNamedArg(DataType, "dtype")], Module
    ]()
else:
    model_factories = ConfigBoundFactoryRegistry()
