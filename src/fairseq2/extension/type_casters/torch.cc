// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/extension/type_casters/torch.h"

#include <stdexcept>

#include "fairseq2/extension/type_casters/torch_api.h"

namespace pybind11 {

template <>
bool
isinstance<at::Tensor>(handle obj)
{
    return PyObject_IsInstance(obj.ptr(), THPVariableClass) != 0;
}

namespace detail {

bool
type_caster<at::Tensor>::load(handle src, bool)
{
    PyObject *ptr = src.ptr();

    if (isinstance<at::Tensor>(ptr)) {
        value = *reinterpret_cast<THPVariable *>(ptr)->cdata;

        return true;
    }

    return false;
}

handle
type_caster<at::Tensor>::cast(const at::Tensor &src, return_value_policy, handle)
{
    return handle{THPVariable_Wrap(src)};
}

bool
type_caster<at::Device>::load(handle src, bool)
{
    PyObject *ptr = src.ptr();

    if (Py_TYPE(ptr) == &THPDeviceType) {
        value = reinterpret_cast<THPDevice *>(ptr)->device;

        return true;
    }

    return false;
}

handle
type_caster<at::Device>::cast(const at::Device &src, return_value_policy, handle)
{
    return handle{THPDevice_New(src)};
}

bool
type_caster<at::ScalarType>::load(handle src, bool)
{
    PyObject *ptr = src.ptr();

    if (Py_TYPE(ptr) == &THPDtypeType) {
        value = reinterpret_cast<THPDtype *>(ptr)->scalar_type;

        return true;
    }

    return false;
}

handle
type_caster<at::ScalarType>::cast(const at::ScalarType &src, return_value_policy, handle)
{
    switch(src) {
    case at::ScalarType::Short:
        return get_dtype("int16");
    case at::ScalarType::Int:
        return get_dtype("int32");
    case at::ScalarType::Long:
        return get_dtype("int64");
    default:
        break;
    }

    throw std::runtime_error{
        "The specified at::ScalarType to torch.dtype conversion is not supported."};
}

handle
type_caster<at::ScalarType>::get_dtype(const char *type_name)
{
    static object torch_module = module_::import("torch");

    object o = torch_module.attr(type_name);

    return o.release();
}

}  // namespace detail
}  // namespace pybind11
