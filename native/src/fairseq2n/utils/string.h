// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <system_error>
#include <type_traits>

namespace fairseq2n {
namespace detail {

template <typename StringViewLike, typename It>
auto
find_first_non_space(It first, It last) noexcept
{
    auto pos = std::find_if_not(first, last, [](int chr)
    {
        return std::isspace(chr);
    });

    return static_cast<typename StringViewLike::size_type>(pos - first);
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

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
T
from_string(std::string_view s, std::int16_t base = 10)
{
    const char *s_end = s.data() + s.size();

    T parsed_value{};

    std::from_chars_result result = std::from_chars(s.data(), s_end, parsed_value, base);
    if (result.ec == std::errc{} && result.ptr == s_end)
        return parsed_value;

    if (result.ec == std::errc::result_out_of_range)
        throw std::out_of_range{"`s` is out of range."};

    throw std::invalid_argument{"`s` does not represent a valid integer value."};
}

}  // namespace fairseq2n
