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

namespace pybind11 {

template <>
bool
isinstance<at::Tensor>(handle obj);

namespace detail {

template <>
struct type_caster<at::Tensor> {
    PYBIND11_TYPE_CASTER(at::Tensor, const_name("torch.Tensor"));

public:
    bool
    load(handle src, bool);

    static handle
    cast(const at::Tensor &src, return_value_policy, handle);
};


template <>
struct type_caster<at::Device> {
    PYBIND11_TYPE_CASTER(at::Device, const_name("torch.device"));

public:
    type_caster() noexcept
      : value{at::kCPU}
    {}

    bool
    load(handle src, bool);

    static handle
    cast(const at::Device &src, return_value_policy, handle);
};

template <>
struct type_caster<at::ScalarType> {
    PYBIND11_TYPE_CASTER(at::ScalarType, const_name("torch.dtype"));

public:
    bool
    load(handle src, bool);

    static handle
    cast(const at::ScalarType &src, return_value_policy, handle);

private:
    static handle
    get_dtype(const char *type_name);
};

}  // namespace detail
}  // namespace pybind11
