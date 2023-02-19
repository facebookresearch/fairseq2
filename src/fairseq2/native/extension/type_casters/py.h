// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

#include <fairseq2/native/py.h>

template <>
struct pybind11::detail::type_caster<fairseq2::py_object> {
    PYBIND11_TYPE_CASTER(fairseq2::py_object, pybind11::detail::const_name("Any"));

public:
    bool
    load(pybind11::handle src, bool)
    {
        value = fairseq2::py_object{src.ptr()};

        return true;
    }

    static pybind11::handle
    cast(const fairseq2::py_object &src, pybind11::return_value_policy, pybind11::handle)
    {
        auto ptr = static_cast<PyObject *>(src.ptr());

        pybind11::handle h{ptr};

        h.inc_ref();

        return h;
    }

    static pybind11::handle
    cast(fairseq2::py_object &&src, pybind11::return_value_policy, pybind11::handle)
    {
        auto ptr = static_cast<PyObject *>(src.release());

        return pybind11::handle{ptr};
    }
};
