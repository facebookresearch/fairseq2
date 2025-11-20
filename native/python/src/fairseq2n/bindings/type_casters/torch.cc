// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/type_casters/torch.h"

#include <torch/version.h>

#include <fairseq2n/exception.h>
#include <fairseq2n/detail/exception.h>

using namespace fairseq2n;
using namespace fairseq2n::detail;

// Taken from <torch/bindings/autograd/python_variable.h>
struct THPVariable {
    PyObject_HEAD
    #if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 10)
    PyObject *THPVariable_Wrap(at::Tensor cdata);
    #else
    PyObject *THPVariable_Wrap(const at::MaybeOwned<at::Tensor> cdata);
    #endif
};

extern PyObject *THPVariableClass;

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 6)
PyObject *THPVariable_Wrap(at::TensorBase var);
#else
PyObject *THPVariable_Wrap(const at::TensorBase &var);
#endif

// Taken from <torch/bindings/Device.h>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPDevice {
    PyObject_HEAD
    at::Device device;
};

extern PyTypeObject THPDeviceType;

PyObject *THPDevice_New(const at::Device &device);

// Taken from <torch/bindings/Dtype.h>
struct THPDtype {
    PyObject_HEAD
    at::ScalarType scalar_type;
};

extern PyTypeObject THPDtypeType;

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
        #if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 10)
        PyObject *THPVariable_Wrap(value = reinterpret_cast<THPVariable *>(ptr)->cdata);
        #else
        PyObject *THPVariable_Wrap(value = const *reinterpret_cast<THPVariable *>(ptr)->cdata);
        #endif
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

    throw_<not_supported_error>(
        "The specified `at::ScalarType` to `torch.dtype` conversion is not supported.");
}

handle
type_caster<at::ScalarType>::get_dtype(const char *type_name)
{
    static object torch_module = module_::import("torch");

    object obj = torch_module.attr(type_name);

    return obj.release();
}

}  // namespace detail
}  // namespace pybind11
