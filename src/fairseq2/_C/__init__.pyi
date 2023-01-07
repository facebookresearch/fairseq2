# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fmt: off

def _supports_cuda() -> bool:
    ...


def _cuda_version() -> tuple[int, int] | None:
    ...
