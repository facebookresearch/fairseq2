# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Type aliases for `torch.device` and `torch.dtype` to make them consistent with
# the standard Python naming convention.
from torch import device as Device
from torch import dtype as DataType

__all__ = ["DataType", "Device"]
