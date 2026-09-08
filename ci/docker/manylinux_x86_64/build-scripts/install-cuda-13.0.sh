#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

curl --location --fail --output cuda.run\
     https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run

sh cuda.run --silent --toolkit --override --no-man-page

rm cuda.run

# We don't need Nsight.
rm -rf /usr/local/cuda-13.0/nsight*

# Add CUDA libraries to the lookup cache of the dynamic linker.
ldconfig
