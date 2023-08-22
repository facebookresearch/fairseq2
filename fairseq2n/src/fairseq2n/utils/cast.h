// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>

#include "fairseq2n/float.h"

namespace fairseq2n::detail {

template <typename Container>
inline constexpr auto
ssize(const Container &container) noexcept
{
    return static_cast<typename Container::difference_type>(container.size());
}

template <typename T, typename U>
inline constexpr T
conditional_cast(U value) noexcept
{
    if constexpr (std::is_same_v<T, U>)
        return value;
    else
        return static_cast<T>(value);
}

template <typename T>
inline constexpr bool
are_equal(const T &lhs, const T &rhs) noexcept
{
    if constexpr (std::is_floating_point_v<T>)
        return are_close(lhs, rhs);
    else
        return lhs == rhs;
}

template <typename T, typename U>
inline constexpr bool
maybe_narrow(U u, T &t) noexcept
{
    if constexpr (std::is_same_v<T, U>) {
        t = u;

        return true;
    } else {
        t = static_cast<T>(u);

        if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
            return are_equal(static_cast<U>(t), u);
        else
            return are_equal(static_cast<U>(t), u) && (t < T{}) == (u < U{});
    }
}

}  // namespace fairseq2n::detail
