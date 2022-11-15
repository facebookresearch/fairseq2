// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <functional>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <ATen/Tensor.h>

#include "fairseq2/native/api.h"
#include "fairseq2/native/utils/memory.h"
#include "fairseq2/native/utils/span.h"

namespace fairseq2 {

class ivariant;

}  // namespace fairseq2

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

using ilist = std::vector<ivariant>;
using idict = std::unordered_map<ivariant, ivariant>;

// A type-safe union shared between Python and native code.
class FAIRSEQ2_API ivariant {
public:
    // uninitialized
    constexpr
    ivariant() noexcept = default;

    constexpr bool
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

    constexpr bool
    is_none() const noexcept
    {
        return std::holds_alternative<none_t>(payload_);
    }

    // boolean
    constexpr
    ivariant(bool value) noexcept
        : payload_{value}
    {}

    constexpr bool
    is_bool() const noexcept
    {
        return std::holds_alternative<bool>(payload_);
    }

    constexpr bool
    as_bool() const
    {
        return std::get<bool>(payload_);
    }

    // int
    constexpr
    ivariant(std::int64_t value) noexcept
        : payload_{value}
    {}

    constexpr bool
    is_int() const noexcept
    {
        return std::holds_alternative<std::int64_t>(payload_);
    }

    constexpr std::int64_t
    as_int() const
    {
        return std::get<std::int64_t>(payload_);
    }

    // double
    constexpr
    ivariant(double value) noexcept
        : payload_{value}
    {}

    constexpr bool
    is_double() const noexcept
    {
        return std::holds_alternative<double>(payload_);
    }

    constexpr double
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

    constexpr bool
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

    constexpr bool
    is_tensor() const noexcept
    {
        return std::holds_alternative<tensor_holder>(payload_);
    }

    at::Tensor &
    as_tensor()
    {
        return std::get<tensor_holder>(payload_).value_;
    }

    const at::Tensor &
    as_tensor() const
    {
        return std::get<tensor_holder>(payload_).value_;
    }

    // ilist
    ivariant(const ilist &value)
        : payload_{std::make_shared<ilist>(value)}
    {}

    ivariant(ilist &&value)
        : payload_{std::make_shared<ilist>(std::move(value))}
    {}

    ivariant(const std::shared_ptr<ilist> &value) noexcept
        : payload_{value}
    {}

    ivariant(std::shared_ptr<ilist> &&value) noexcept
        : payload_{std::move(value)}
    {}

    constexpr bool
    is_list() const noexcept
    {
        return std::holds_alternative<std::shared_ptr<ilist>>(payload_);
    }

    ilist &
    as_list()
    {
        return *std::get<std::shared_ptr<ilist>>(payload_);
    }

    const ilist &
    as_list() const
    {
        return *std::get<std::shared_ptr<ilist>>(payload_);
    }

    std::shared_ptr<ilist> &
    as_shared_list()
    {
        return std::get<std::shared_ptr<ilist>>(payload_);
    }

    std::shared_ptr<const ilist>
    as_shared_list() const
    {
        return std::get<std::shared_ptr<ilist>>(payload_);
    }

    // idict
    ivariant(const idict &value)
        : payload_{std::make_shared<idict>(value)}
    {}

    ivariant(idict &&value)
        : payload_{std::make_shared<idict>(std::move(value))}
    {}

    ivariant(const std::shared_ptr<idict> &value) noexcept
        : payload_{value}
    {}

    ivariant(std::shared_ptr<idict> &&value) noexcept
        : payload_{std::move(value)}
    {}

    constexpr bool
    is_dict() const noexcept
    {
        return std::holds_alternative<std::shared_ptr<idict>>(payload_);
    }

    idict &
    as_dict()
    {
        return *std::get<std::shared_ptr<idict>>(payload_);
    }

    const idict &
    as_dict() const
    {
        return *std::get<std::shared_ptr<idict>>(payload_);
    }

    std::shared_ptr<idict> &
    as_shared_dict()
    {
        return std::get<std::shared_ptr<idict>>(payload_);
    }

    std::shared_ptr<const idict>
    as_shared_dict() const
    {
        return std::get<std::shared_ptr<idict>>(payload_);
    }

public:
    friend FAIRSEQ2_API bool
    operator==(const ivariant &lhs, const ivariant &rhs);

    friend bool
    operator!=(const ivariant &lhs, const ivariant &rhs)
    {
        return !(lhs == rhs);
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
                 tensor_holder,
                 std::shared_ptr<ilist>,
                 std::shared_ptr<idict>> payload_{};
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
