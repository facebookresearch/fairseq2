#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

function print_usage
{
    echo "Usage: run-shellcheck PATHNAME"
}

function exit_with_usage
{
    print_usage >&1

    exit 0
}

function exit_with_error
{
    print_usage >&2

    exit 1
}


if [[ $# -ne 1 ]]; then
    exit_with_error
fi

if [[ $1 == -h || $1 == --help ]]; then
    exit_with_usage
fi

find "$1" \( -type d \( -name '.?*' -or -name 'build' -or -name 'third-party' \) -prune \) -or\
    \( -type f -name '*.sh' -print0 \) |
        xargs --no-run-if-empty --null shellcheck -f gcc --severity=warning --norc
