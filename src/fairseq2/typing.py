# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["DataType", "Device"]


import torch.optim.lr_scheduler

# Type aliases for `torch.device` and `torch.dtype` to make them consistent with
# the standard Python naming convention.
from torch import device as Device, dtype as DataType

# LRScheduler changed visibility around pytorch 13.1
if hasattr(torch.optim.lr_scheduler, "LRScheduler"):
    LRScheduler = torch.optim.lr_scheduler.LRScheduler  # type: ignore
else:
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # type: ignore
