// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cctype>

namespace fairseq2 {
namespace detail {

template <typename StringViewLike, typename It>
auto
find_first_non_space(It first, It last) noexcept
{
    auto iter = std::find_if_not(first, last, [](int c) {
        return std::isspace(c);
    });

    return static_cast<typename StringViewLike::size_type>(iter - first);
}

}  // namespace detail

template <typename StringViewLike>
inline auto
ltrim(const StringViewLike &s) noexcept
{
    auto offset = detail::find_first_non_space<StringViewLike>(s.begin(), s.end());

    return s.remove_prefix(offset);
}

template <typename StringViewLike>
inline auto
rtrim(const StringViewLike &s) noexcept
{
    auto offset = detail::find_first_non_space<StringViewLike>(s.rbegin(), s.rend());

    return s.remove_suffix(offset);
}

template <typename StringViewLike>
inline auto
trim(const StringViewLike &s) noexcept
{
    StringViewLike t = ltrim(s);

    return rtrim(t);
}

}  // namespace fairseq2
