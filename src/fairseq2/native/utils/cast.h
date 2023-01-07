// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <type_traits>

namespace fairseq2::detail {

template<typename Container>
inline constexpr auto
ssize(const Container &c) noexcept
{
    return static_cast<typename Container::difference_type>(c.size());
}

template <typename T, typename U>
inline constexpr bool
try_narrow(U u, T &t) noexcept
{
    if constexpr (std::is_same_v<T, U>) {
        t = u;

        return true;
    } else {
        t = static_cast<T>(u);

        if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
            return static_cast<U>(t) == u;
        else
            return static_cast<U>(t) == u && (t < T{}) == (u < U{});
    }
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

}  // namespace fairseq2::detail
