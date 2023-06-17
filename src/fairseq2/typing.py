# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import device, dtype
from typing_extensions import TypeAlias

# Type aliases for `torch.device` and `torch.dtype` to make them consistent with
# the regular Python naming convention.
Device: TypeAlias = device

DataType: TypeAlias = dtype
