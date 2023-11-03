#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

for variant in cpu cu116 cu117 cu118 cu121; do
    docker build --network host --tag ghcr.io/facebookresearch/fairseq2-ci-manylinux_x86_64:2-$variant -f manylinux_x86_64/Dockerfile.$variant manylinux_x86_64
done

for variant in cpu cu116 cu117 cu118 cu121; do
    echo docker push ghcr.io/facebookresearch/fairseq2-ci-manylinux_x86_64:2-$variant
done

echo docker logout ghcr.io
