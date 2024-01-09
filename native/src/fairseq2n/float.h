// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace fairseq2n {

using float32 = float;
using float64 = double;

namespace detail {

template <typename T>
struct rel {};

template <>
struct rel<float32> {
    static constexpr float32 value = 0.0001F;
};

template <>
struct rel<float64> {
    static constexpr float64 value = 0.0001;
};

}  // namespace detail

// `T` must be a floating-point type.
template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
inline constexpr bool
are_close(T lhs, T rhs, T rel = detail::rel<T>::value) noexcept
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    if (lhs == rhs)
        return true;
#pragma GCC diagnostic pop

    return std::abs(rhs - lhs) < rel * std::max(std::abs(lhs), std::abs(rhs));
}

}  // namespace fairseq2n
