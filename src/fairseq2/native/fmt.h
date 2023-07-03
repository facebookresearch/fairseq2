// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <type_traits>

#include <fmt/core.h>
#include <fmt/format.h>

template <typename T>
struct fmt::formatter<
    T, std::enable_if_t<
        // `T` must have a `fairseq2::repr()` overload.
        std::is_same_v<decltype(fairseq2::repr(std::declval<T>())), std::string>, char>>
  : fmt::formatter<std::string> {

    auto
    format(const T &t, format_context &ctx) const
    {
        return formatter<std::string>::format(fairseq2::repr(t), ctx);
    }
};
