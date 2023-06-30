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

inline std::string_view
remove_prefix(std::string_view s, std::string_view::size_type n) noexcept
{
    auto t = s;

    t.remove_prefix(n);

    return t;
}

inline std::string_view
remove_suffix(std::string_view s, std::string_view::size_type n) noexcept
{
    auto t = s;

    t.remove_suffix(n);

    return t;
}

template <typename StringViewLike>
inline auto
ltrim(const StringViewLike &s) noexcept
{
    auto b = s.begin();
    auto e = s.end();

    auto offset = detail::find_first_non_space<StringViewLike>(b, e);

    return remove_prefix(s, offset);
}

template <typename StringViewLike>
inline auto
rtrim(const StringViewLike &s) noexcept
{
    auto b = s.rbegin();
    auto e = s.rend();

    auto offset = detail::find_first_non_space<StringViewLike>(b, e);

    return remove_suffix(s, offset);
}

template <typename StringViewLike>
inline auto
trim(const StringViewLike &s) noexcept
{
    return rtrim(ltrim(s));
}

}  // namespace fairseq2
