// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cerrno>
#include <system_error>

namespace fairseq2n::detail {

inline std::error_code
last_error() noexcept
{
    return std::error_code{errno, std::generic_category()};
}

}  // namespace fairseq2n::detail
