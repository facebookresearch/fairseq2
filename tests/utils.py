# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import Generator

import torch

from fairseq2.typing import Device


@contextmanager
def tmp_rng_seed(device: Device, seed: int = 0) -> Generator[None, None, None]:
    """Sets a temporary manual RNG seed.

    The RNG is reset to its original state once the block is exited.
    """
    device = torch.device(device)

    if device.type == "cuda":
        devices = [device]
    else:
        devices = []

    with torch.random.fork_rng(devices):
        torch.manual_seed(seed)

        yield
