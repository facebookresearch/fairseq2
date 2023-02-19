// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <ATen/Device.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>

template <>
bool
pybind11::isinstance<at::Tensor>(pybind11::handle obj);

template <>
struct pybind11::detail::type_caster<at::Tensor> {
    PYBIND11_TYPE_CASTER(at::Tensor, pybind11::detail::const_name("torch.Tensor"));

public:
    bool
    load(pybind11::handle src, bool);

    static pybind11::handle
    cast(const at::Tensor &src, pybind11::return_value_policy, pybind11::handle);
};


template <>
struct pybind11::detail::type_caster<at::Device> {
    PYBIND11_TYPE_CASTER(at::Device, pybind11::detail::const_name("torch.device"));

public:
    type_caster() noexcept
        : value{at::kCPU}
    {}

    bool
    load(pybind11::handle src, bool);

    static pybind11::handle
    cast(const at::Device &src, pybind11::return_value_policy, pybind11::handle);
};

template <>
struct pybind11::detail::type_caster<at::ScalarType> {
    PYBIND11_TYPE_CASTER(at::ScalarType, pybind11::detail::const_name("torch.dtype"));

public:
    bool
    load(pybind11::handle src, bool);

    static pybind11::handle
    cast(const at::ScalarType &src, pybind11::return_value_policy, pybind11::handle);

private:
    static pybind11::handle
    get_dtype(const char *type_name);
};
