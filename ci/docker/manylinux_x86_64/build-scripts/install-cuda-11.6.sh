#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

curl --location --fail --output cuda.run\
    https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run

sh cuda.run --silent --toolkit --override --no-man-page

rm cuda.run

# We don't need Nsight.
rm -rf /usr/local/cuda-11.6/nsight*

# Add CUDA libraries to the lookup cache of the dynamic linker.
ldconfig
