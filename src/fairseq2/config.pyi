# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

def get_cuda_version() -> Optional[Tuple[int, int]]: ...
def supports_cuda() -> bool: ...
