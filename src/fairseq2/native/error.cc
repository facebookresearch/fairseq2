// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/error.h"

#include <cstdio>
#include <cstdlib>

#include <fmt/core.h>

namespace fairseq2::detail {

#ifndef NDEBUG

void
unreachable(const char *file, int line)
{
    fmt::print(stderr, "Unreachable code executed at {}:{}.\n", file, line);

    std::fflush(stderr);  // NOLINT(cert-err33-c)

    std::abort();
}

#endif

}  // namespace fairseq2::detail
