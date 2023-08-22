// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/util/order_preserving_flat_hash_map.h>

#include "fairseq2n/api.h"
#include "fairseq2n/float.h"
#include "fairseq2n/fmt.h"
#include "fairseq2n/memory.h"
#include "fairseq2n/data/immutable_string.h"
#include "fairseq2n/data/py.h"
#include "fairseq2n/utils/cast.h"

namespace fairseq2n {

template <
    typename Key,
    typename T,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    typename Allocator = std::allocator<std::pair<Key, T>>>
using flat_hash_map = ska_ordered::order_preserving_flat_hash_map<
    Key, T, Hash, KeyEqual, Allocator>;

enum class data_type : std::int16_t {
    bool_,
    int_,
    float_,
    string,
    tensor,
    memory_block,
    list,
    dict,
    pyobj,
};

class data {
public:
    data() noexcept = default;

    data(const data &) = default;
    data &operator=(const data &) = default;

    data(data &&other) noexcept = default;
    data &operator=(data &&other) noexcept = default;

   ~data() = default;

    data_type
    type() const noexcept
    {
        return data_type{detail::conditional_cast<std::int16_t>(payload_.index())};
    }

    // bool
    data(bool value) noexcept
        : payload_{value}
    {}

    bool
    is_bool() const noexcept
    {
        return is<bool>();
    }

    bool
    as_bool() const noexcept
    {
        return as<bool>();
    }

    // int
    data(std::int64_t value) noexcept
        : payload_{value}
    {}

    bool
    is_int() const noexcept
    {
        return is<std::int64_t>();
    }

    std::int64_t
    as_int() const noexcept
    {
        return as<std::int64_t>();
    }

    // float
    data(float64 value) noexcept
        : payload_{value}
    {}

    bool
    is_float() const noexcept
    {
        return is<float64>();
    }

    float64
    as_float() const noexcept
    {
        return as<float64>();
    }

    // string
    data(const char *value)
      : payload_{immutable_string{value}}
    {}

    data(const std::string &value)
      : payload_{immutable_string{value}}
    {}

    data(std::string_view value)
      : payload_{immutable_string{value}}
    {}

    data(const immutable_string &value) noexcept
      : payload_{value}
    {}

    data(immutable_string &&value) noexcept
      : payload_{std::move(value)}
    {}

    bool
    is_string() const noexcept
    {
        return is<immutable_string>();
    }

    immutable_string &
    as_string() & noexcept
    {
        return as<immutable_string>();
    }

    const immutable_string &
    as_string() const & noexcept
    {
        return as<immutable_string>();
    }

    immutable_string &&
    as_string() && noexcept
    {
        return move_as<immutable_string>();
    }

    // Tensor
    data(const at::Tensor &value)
      : payload_{value}
    {}

    data(at::Tensor &&value) noexcept
      : payload_{std::move(value)}
    {}

    bool
    is_tensor() const noexcept
    {
        return is<at::Tensor>();
    }

    at::Tensor &
    as_tensor() & noexcept
    {
        return as<at::Tensor>();
    }

    const at::Tensor &
    as_tensor() const & noexcept
    {
        return as<at::Tensor>();
    }

    at::Tensor &&
    as_tensor() && noexcept
    {
        return move_as<at::Tensor>();
    }

    // memory_block
    data(const memory_block &value) noexcept
      : payload_{value}
    {}

    data(memory_block &&value) noexcept
      : payload_{std::move(value)}
    {}

    bool
    is_memory_block() const noexcept
    {
        return is<memory_block>();
    }

    const memory_block &
    as_memory_block() const & noexcept
    {
        return as<memory_block>();
    }

    memory_block &&
    as_memory_block() && noexcept
    {
        return move_as<memory_block>();
    }

    // list
    data(const std::vector<data> &value)
      : payload_{value}
    {}

    data(std::vector<data> &&value) noexcept
      : payload_{std::move(value)}
    {}

    bool
    is_list() const noexcept
    {
        return is<std::vector<data>>();
    }

    std::vector<data> &
    as_list() & noexcept
    {
        return as<std::vector<data>>();
    }

    const std::vector<data> &
    as_list() const & noexcept
    {
        return as<std::vector<data>>();
    }

    std::vector<data> &&
    as_list() && noexcept
    {
        return move_as<std::vector<data>>();
    }

    // dict
    data(const flat_hash_map<std::string, data> &value)
      : payload_{value}
    {}

    data(flat_hash_map<std::string, data> &&value) noexcept
      : payload_{std::move(value)}
    {}

    bool
    is_dict() const noexcept
    {
        return is<flat_hash_map<std::string, data>>();
    }

    flat_hash_map<std::string, data> &
    as_dict() & noexcept
    {
        return as<flat_hash_map<std::string, data>>();
    }

    const flat_hash_map<std::string, data> &
    as_dict() const & noexcept
    {
        return as<flat_hash_map<std::string, data>>();
    }

    flat_hash_map<std::string, data> &&
    as_dict() && noexcept
    {
        return move_as<flat_hash_map<std::string, data>>();
    }

    // py_object
    data(const py_object &value) noexcept
      : payload_{value}
    {}

    data(py_object &&value) noexcept
      : payload_{std::move(value)}
    {}

    bool
    is_py() const noexcept
    {
        return is<py_object>();
    }

    py_object &
    as_py() & noexcept
    {
        return as<py_object>();
    }

    const py_object &
    as_py() const & noexcept
    {
        return as<py_object>();
    }

    py_object &&
    as_py() && noexcept
    {
        return move_as<py_object>();
    }

private:
    template <typename T>
    bool
    is() const noexcept
    {
        return std::holds_alternative<T>(payload_);
    }

    template <typename T>
    T &
    as() noexcept
    {
        return *std::get_if<T>(&payload_);
    }

    template <typename T>
    const T &
    as() const noexcept
    {
        return *std::get_if<T>(&payload_);
    }

    template <typename T>
    T &&
    move_as() noexcept
    {
        return std::move(as<T>());
    }

private:
    std::variant<
        bool,
        std::int64_t,
        float64,
        immutable_string,
        at::Tensor,
        memory_block,
        std::vector<data>,
        flat_hash_map<std::string, data>,
        py_object> payload_{};
};

using data_list = std::vector<data>;

using data_dict = flat_hash_map<std::string, data>;

template <>
struct FAIRSEQ2_API repr<data_type> {
    std::string
    operator()(data_type dt) const;
};

}  // namespace fairseq2n
