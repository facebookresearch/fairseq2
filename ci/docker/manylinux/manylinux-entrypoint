#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Developer Toolset 10 is already enabled on manylinux2014.
if [[ $DEVTOOLSET -ne 10 ]]; then
    source scl_source enable devtoolset-$DEVTOOLSET
fi

exec "$@"
