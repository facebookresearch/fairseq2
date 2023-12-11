#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

function print_usage
{
    echo "Usage: set-version PEP440_VERSION"
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

function replace_match
{
    sed --in-place --expression "$2" "$1"
}

function extract_pep_version
{
    echo "$1"
}

function extract_mmm_version
{
    # Grep major, minor, and micro segments.
    echo "$1" | grep --only-matching --extended-regexp '^([0-9]+\.)*[0-9]+' -
}

if [[ $# -ne 1 ]]; then
    exit_with_error
fi

if [[ $1 == -h || $1 == --help ]]; then
    exit_with_usage
fi

pep_ver=$(extract_pep_version "$1")
mmm_ver=$(extract_mmm_version "$1")

base=$(cd "$(dirname "$0")/.." && pwd)

# Update Python distribution.
replace_match\
    "$base/setup.py"\
    "s/^version = \".*\"$/version = \"$pep_ver\"/"

# Update Python package.
replace_match\
    "$base/src/fairseq2/__init__.py"\
    "s/^__version__ = \".*\"$/__version__ = \"$pep_ver\"/"

# Update fairseq2n CMake project.
replace_match\
    "$base/native/CMakeLists.txt"\
    "s/VERSION .* LANGUAGES/VERSION $mmm_ver LANGUAGES/"

# Update fairseq2n Python distribution.
replace_match\
    "$base/native/python/setup.py"\
    "s/    version=\".*\",$/    version=\"$pep_ver\",/"

# Update fairseq2n Python package.
replace_match\
    "$base/native/python/src/fairseq2n/__init__.py"\
    "s/^__version__ = \".*\"$/__version__ = \"$pep_ver\"/"

# Update VERSION file.
echo "$pep_ver" > "$base/VERSION"
