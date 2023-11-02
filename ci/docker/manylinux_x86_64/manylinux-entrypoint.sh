#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

case $VARIANT in
cu116)
    PATH=/usr/local/cuda-11.6/bin:$PATH

    LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
    ;;
cu117)
    PATH=/usr/local/cuda-11.7/bin:$PATH

    LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
    ;;
cu118)
    PATH=/usr/local/cuda-11.8/bin:$PATH

    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    ;;
cu121)
    PATH=/usr/local/cuda-12.1/bin:$PATH

    LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
    ;;
esac

exec "$@"
