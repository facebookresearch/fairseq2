// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace fairseq2n {

template <typename T>
struct repr {};  // disabled (i.e. poisoned)

}  // namespace fairseq2n


// Allow the use of this header file even if libfmt is not available.
#if __has_include(<fmt/core.h>)

#include <string>
#include <type_traits>

#include <fmt/core.h>
#include <fmt/format.h>

template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_invocable_v<fairseq2n::repr<T>, T>, char>>
  : fmt::formatter<std::string> {

    auto
    format(const T &t, format_context &ctx) const
    {
        return formatter<std::string>::format(fairseq2n::repr<T>{}(t), ctx);
    }
};

#endif
