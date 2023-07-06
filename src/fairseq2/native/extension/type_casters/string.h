// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <string_view>

#include <pybind11/pybind11.h>

namespace pybind11::detail {

template <>
struct type_caster<std::string> {
    PYBIND11_TYPE_CASTER(std::string, const_name("str"));

private:
    using inner_caster = string_caster<std::string, false>;

public:
    bool
    load(handle src, bool convert);

    static handle
    cast(const std::string &src, return_value_policy policy, handle parent)
    {
        return inner_caster::cast(src, policy, parent);
    }

private:
    inner_caster inner_caster_{};
};

template <>
struct type_caster<std::string_view> {
    PYBIND11_TYPE_CASTER(std::string_view, const_name("str"));

private:
    using inner_caster = string_caster<std::string_view, true>;

public:
    bool
    load(handle src, bool convert);

    static handle
    cast(const std::string_view &src, return_value_policy policy, handle parent)
    {
        return inner_caster::cast(src, policy, parent);
    }

private:
    inner_caster inner_caster_{};
};

}  // namespace pybind11::detail
