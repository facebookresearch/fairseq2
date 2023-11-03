#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

repo=ghcr.io/facebookresearch

arch=x86_64

version=2

declare -a variants=(cpu cu116 cu117 cu118 cu121)

for variant in "${variants[@]}"; do
    docker build\
        --network host\
        --tag $repo/fairseq2-ci-manylinux_$arch:$version-$variant\
        --file manylinux_$arch/Dockerfile.$variant\
        manylinux_$arch/
done

for variant in "${variants[@]}"; do
    docker push $repo/fairseq2-ci-manylinux_$arch:$version-$variant
done

docker logout ghcr.io
