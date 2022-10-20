# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# We import `torch` to make sure that libtorch.so is loaded into the process
# before our extension modules.
import torch  # noqa: F401

__version__ = "0.1.0.dev0"
