// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// Taken from <torch/csrc/autograd/python_variable.h>
struct THPVariable {
    PyObject_HEAD
    at::MaybeOwned<at::Tensor> cdata;
};

extern PyObject *THPVariableClass;

PyObject *THPVariable_Wrap(at::TensorBase var);

// Taken from <torch/csrc/Device.h>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPDevice {
    PyObject_HEAD
    at::Device device;
};

extern PyTypeObject THPDeviceType;

PyObject *THPDevice_New(const at::Device &device);

// Taken from <torch/csrc/Dtype.h>
struct THPDtype {
    PyObject_HEAD
    at::ScalarType scalar_type;
};

extern PyTypeObject THPDtypeType;
