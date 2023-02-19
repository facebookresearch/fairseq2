// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <string_view>

#include <pybind11/pybind11.h>

template <>
struct pybind11::detail::type_caster<std::string> {
    PYBIND11_TYPE_CASTER(std::string, pybind11::detail::const_name("str"));

private:
    using subcaster = pybind11::detail::string_caster<std::string, false>;

public:
    bool
    load(pybind11::handle src, bool convert);

    static pybind11::handle
    cast(const std::string &src, pybind11::return_value_policy policy, pybind11::handle parent)
    {
        return subcaster::cast(src, policy, parent);
    }

private:
    subcaster subcaster_{};
};

template <>
struct pybind11::detail::type_caster<std::string_view> {
    PYBIND11_TYPE_CASTER(std::string_view, pybind11::detail::const_name("str"));

private:
    using subcaster = pybind11::detail::string_caster<std::string_view, true>;

public:
    bool
    load(pybind11::handle src, bool convert);

    static pybind11::handle
    cast(const std::string_view &src, pybind11::return_value_policy policy, pybind11::handle parent)
    {
        return subcaster::cast(src, policy, parent);
    }

private:
    subcaster subcaster_{};
};
