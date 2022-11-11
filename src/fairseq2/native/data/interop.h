// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <functional>
#include <string_view>
#include <utility>
#include <variant>

#include <ATen/Tensor.h>

#include "fairseq2/native/api.h"
#include "fairseq2/native/utils/memory.h"
#include "fairseq2/native/utils/span.h"

namespace fairseq2 {

// An immutable UTF-8 string type that supports zero-copy marshalling between
// Python and native code.
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

// A type-safe union to be shared between Python and native code.
class FAIRSEQ2_API ivariant {
public:
    // uninitialized
    ivariant() noexcept = default;

    bool
    is_uninitialized() const noexcept
    {
        return std::holds_alternative<uninitialized_t>(payload_);
    }

    // none
    static ivariant
    none()
    {
        ivariant v{};

        v.payload_ = none_t{};

        return v;
    }

    bool
    is_none() const noexcept
    {
        return std::holds_alternative<none_t>(payload_);
    }

    // boolean
    ivariant(bool value) noexcept
        : payload_{value}
    {}

    bool
    is_bool() const noexcept
    {
        return std::holds_alternative<bool>(payload_);
    }

    bool
    as_bool() const
    {
        return std::get<bool>(payload_);
    }

    // int
    ivariant(std::int64_t value) noexcept
        : payload_{value}
    {}

    bool
    is_int() const noexcept
    {
        return std::holds_alternative<std::int64_t>(payload_);
    }

    std::int64_t
    as_int() const
    {
        return std::get<std::int64_t>(payload_);
    }

    // double
    ivariant(double value) noexcept
        : payload_{value}
    {}

    bool
    is_double() const noexcept
    {
        return std::holds_alternative<double>(payload_);
    }

    double
    as_double() const
    {
        return std::get<double>(payload_);
    }

    // string
    ivariant(const char *value)
        : payload_{value}
    {}

    ivariant(const std::string &value)
        : payload_{value}
    {}

    ivariant(std::string_view value)
        : payload_{value}
    {}

    ivariant(const istring &value) noexcept
        : payload_{value}
    {}

    ivariant(istring &&value) noexcept
        : payload_{std::move(value)}
    {}

    bool
    is_string() const noexcept
    {
        return std::holds_alternative<istring>(payload_);
    }

    istring &
    as_string()
    {
        return std::get<istring>(payload_);
    }

    const istring &
    as_string() const
    {
        return std::get<istring>(payload_);
    }

    // tensor
    ivariant(const at::Tensor &value)
        : payload_{value}
    {}

    ivariant(at::Tensor &&value)
        : payload_{std::move(value)}
    {}

    bool
    is_tensor() const noexcept
    {
        return std::holds_alternative<tensor_holder>(payload_);
    }

    at::Tensor
    as_tensor()
    {
        return std::get<tensor_holder>(payload_).value_;
    }

    const at::Tensor &
    as_tensor() const
    {
        return std::get<tensor_holder>(payload_).value_;
    }

public:
    friend constexpr bool
    operator==(const ivariant &lhs, const ivariant &rhs)
    {
        return lhs.payload_ == rhs.payload_;
    }

    friend constexpr bool
    operator!=(const ivariant &lhs, const ivariant &rhs)
    {
        return lhs.payload_ != rhs.payload_;
    }

private:
    struct uninitialized_t {
        friend constexpr bool
        operator==(const uninitialized_t &, const uninitialized_t &) noexcept
        {
            return false;
        }

        friend constexpr bool
        operator!=(const uninitialized_t &, const uninitialized_t &) noexcept
        {
            return true;
        }
    };

    struct none_t {
        friend constexpr bool
        operator==(const none_t &, const none_t &) noexcept
        {
            return true;
        }

        friend constexpr bool
        operator!=(const none_t &, const none_t &) noexcept
        {
            return false;
        }
    };

    struct tensor_holder {
        tensor_holder(const at::Tensor &value) noexcept
            : value_{value}
        {}

        tensor_holder(at::Tensor &&value) noexcept
            : value_{std::move(value)}
        {}

        friend bool
        operator==(const tensor_holder &lhs, const tensor_holder &rhs) noexcept
        {
            return lhs.value_.unsafeGetTensorImpl() == rhs.value_.unsafeGetTensorImpl();
        }

        friend bool
        operator!=(const tensor_holder &lhs, const tensor_holder &rhs) noexcept
        {
            return lhs.value_.unsafeGetTensorImpl() != rhs.value_.unsafeGetTensorImpl();
        }

        at::Tensor value_;
    };

    std::variant<uninitialized_t,
                 none_t,
                 bool,
                 std::int64_t,
                 double,
                 istring,
                 tensor_holder> payload_;
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

template <>
struct FAIRSEQ2_API std::hash<fairseq2::ivariant> {
    std::size_t
    operator()(const fairseq2::ivariant &value) const;

private:
    template <typename T>
    inline std::size_t
    get_hash(const T &value) const
    {
        return std::hash<T>{}(value);
    }
};
