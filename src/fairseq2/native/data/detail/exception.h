// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <exception>
#include <optional>
#include <utility>

#include <fmt/core.h>

#include "fairseq2/native/data/data.h"
#include "fairseq2/native/data/data_pipeline.h"

namespace fairseq2::detail {

template <typename... Args>
[[noreturn]] inline void
throw_data_pipeline_error(
    std::optional<data> example, fmt::format_string<Args...> format, Args &&...args)
{
    throw data_pipeline_error{
        fmt::vformat(format, fmt::make_format_args(args...)), std::move(example)};
}

template <typename... Args>
[[noreturn]] inline void
throw_data_pipeline_error_with_nested(
    std::optional<data> example, fmt::format_string<Args...> format, Args &&...args)
{
    std::throw_with_nested(data_pipeline_error{
        fmt::vformat(format, fmt::make_format_args(args...)), std::move(example)});
}

}  // namespace fairseq2::detail
