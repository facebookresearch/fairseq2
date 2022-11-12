# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# fmt: off

def supports_cuda() -> bool:
    """Indicates whether the library supports CUDA.

    :returns:
        A boolean value indicating whether the library supports CUDA.
    """

def get_cuda_version() -> tuple[int, int] | None:
    """Returns the version of CUDA with which the library was built.

    :returns:
        The major and minor version segments.
    """
