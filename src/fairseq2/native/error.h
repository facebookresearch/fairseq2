// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cerrno>
#include <system_error>

namespace fairseq2::detail {

inline std::error_code
last_error() noexcept
{
    return std::error_code{errno, std::generic_category()};
}

#ifdef NDEBUG

[[noreturn]] inline void
unreachable()
{
    __builtin_unreachable();
}

#else

[[noreturn]] void
unreachable(const char *file = __builtin_FILE(), int line = __builtin_LINE());

#endif

}  // namespace fairseq2::detail
