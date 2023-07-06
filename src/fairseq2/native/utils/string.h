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
    auto iter = std::find_if_not(first, last, [](int chr)
    {
        return std::isspace(chr);
    });

    return static_cast<typename StringViewLike::size_type>(iter - first);
}

}  // namespace detail

inline std::string_view
remove_prefix(std::string_view s, std::string_view::size_type n) noexcept
{
    auto tmp = s;

    tmp.remove_prefix(n);

    return tmp;
}

inline std::string_view
remove_suffix(std::string_view s, std::string_view::size_type n) noexcept
{
    auto tmp = s;

    tmp.remove_suffix(n);

    return tmp;
}

template <typename StringViewLike>
inline auto
ltrim(const StringViewLike &s) noexcept
{
    auto begin = s.begin();
    auto end = s.end();

    auto offset = detail::find_first_non_space<StringViewLike>(begin, end);

    return remove_prefix(s, offset);
}

template <typename StringViewLike>
inline auto
rtrim(const StringViewLike &s) noexcept
{
    auto begin = s.rbegin();
    auto end = s.rend();

    auto offset = detail::find_first_non_space<StringViewLike>(begin, end);

    return remove_suffix(s, offset);
}

template <typename StringViewLike>
inline auto
trim(const StringViewLike &s) noexcept
{
    return rtrim(ltrim(s));
}

}  // namespace fairseq2
