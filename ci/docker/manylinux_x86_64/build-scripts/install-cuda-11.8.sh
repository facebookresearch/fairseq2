#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

curl --location --fail --output cuda.run\
    https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

sh cuda.run --silent --toolkit --override --no-man-page

rm cuda.run

# We don't need Nsight.
rm -rf /usr/local/cuda-11.8/nsight*

# Add CUDA libraries to the lookup cache of the dynamic linker.
ldconfig
