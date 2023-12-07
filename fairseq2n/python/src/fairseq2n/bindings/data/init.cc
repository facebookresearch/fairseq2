// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/module.h"

#include <fairseq2n/data/py.h>

namespace py = pybind11;

namespace fairseq2n {
namespace {

void
inc_ref(py_object &obj) noexcept  // NOLINT(bugprone-exception-escape)
{
    py::gil_scoped_acquire gil{};

    Py_IncRef(static_cast<PyObject *>(obj.ptr()));
}

void
dec_ref(py_object &obj) noexcept // NOLINT(bugprone-exception-escape)
{
    py::gil_scoped_acquire gil{};

    Py_DecRef(static_cast<PyObject *>(obj.ptr()));
}

}  // namespace

void
def_data(py::module_ &base)
{
    detail::register_py_interpreter(inc_ref, dec_ref);

    py::module_ m = base.def_submodule("data");

    def_audio(m);

    def_image(m);

    def_data_pipeline(m);

    def_string(m);

    def_text(m);
}

}  // namespace fairseq2n
