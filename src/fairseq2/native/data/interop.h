// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

#include "fairseq2/native/api.h"
#include "fairseq2/native/utils/memory.h"
#include "fairseq2/native/utils/span.h"

namespace fairseq2 {

class FAIRSEQ2_API istring {
public:
    using value_type      = const char;
    using size_type       = span<const char>::size_type;
    using difference_type = span<const char>::difference_type;
    using reference       = span<const char>::reference;
    using const_reference = span<const char>::const_reference;
    using pointer         = span<const char>::pointer;
    using const_pointer   = span<const char>::const_pointer;
    using iterator        = span<const char>::iterator;
    using const_iterator  = span<const char>::const_iterator;

    constexpr
    istring() noexcept = default;

    istring(const char *s)
        : istring{std::string_view{s}}
    {}

    istring(const std::string &s)
        : istring{static_cast<std::string_view>(s)}
    {}

    istring(std::string_view s);

    constexpr const_iterator
    begin() const noexcept
    {
        return chars().begin();
    }

    constexpr const_iterator
    end() const noexcept
    {
        return chars().end();
    }

    constexpr const_reference
    operator[](size_type pos) const
    {
        return chars()[pos];
    }

    constexpr const_pointer
    data() const noexcept
    {
        return chars().data();
    }

    constexpr size_type
    size() const noexcept
    {
        return chars().size();
    }

    constexpr bool
    empty() const noexcept
    {
        return chars().empty();
    }

    friend constexpr bool
    operator==(const istring &lhs, const istring &rhs) noexcept
    {
        return lhs.view() == rhs.view();
    }

    friend constexpr bool
    operator!=(const istring &lhs, const istring &rhs) noexcept
    {
        return lhs.view() != rhs.view();
    }

    const_pointer
    c_str() const noexcept;

    constexpr
    operator std::string_view() const noexcept
    {
        return view();
    }

    std::string
    to_string() const
    {
        return {data(), size()};
    }

    std::size_t
    get_code_point_length() const;

private:
    constexpr std::string_view
    view() const noexcept
    {
        return {data(), size()};
    }

    constexpr span<value_type>
    chars() const noexcept
    {
        return trim_null(bits_.cast<value_type>());
    }

    static constexpr span<value_type>
    trim_null(span<value_type> c) noexcept
    {
        if (c.empty())
            return c;
        else
            return c.first(c.size() - 1);
    }

private:
    detail::memory_block bits_{};
};

}  // namespace fairseq2

template <>
struct std::hash<fairseq2::istring> {
    inline std::size_t
    operator()(const fairseq2::istring &value) const noexcept
    {
        return std::hash<std::string_view>{}(value);
    }
};
