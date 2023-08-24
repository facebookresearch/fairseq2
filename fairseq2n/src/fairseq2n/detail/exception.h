// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <exception>
#include <system_error>
#include <utility>

#include <fmt/core.h>

namespace fairseq2n::detail {

template <typename E, typename... Args>
[[noreturn]] inline void
throw_(fmt::format_string<Args...> format, Args &&...args)
{
    throw E{fmt::vformat(format, fmt::make_format_args(args...))};
}

template <typename E, typename... Args>
[[noreturn]] inline void
throw_with_nested(fmt::format_string<Args...> format, Args &&...args)
{
    std::throw_with_nested(E{fmt::vformat(format, fmt::make_format_args(args...))});
}

template <typename... Args>
[[noreturn]] inline void
throw_system_error(std::error_code ec, fmt::format_string<Args...> format, Args &&...args)
{
    throw std::system_error{ec, fmt::vformat(format, fmt::make_format_args(args...))};
}

}  // namespace fairseq2n::detail
