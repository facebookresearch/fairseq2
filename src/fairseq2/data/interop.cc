// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <string_view>

#include <fmt/core.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "fairseq2/native/data/interop.h"

namespace py = pybind11;

using fairseq2::istring;

PYBIND11_MODULE(interop, m)
{
    py::options opts{};
    opts.disable_function_signatures();

    // istring
    py::class_<istring>(m, "IString",
R"docstr(
    An immutable UTF-8 string type that supports zero-copy marshalling between
    Python and native code.
)docstr")
        .def(py::init<>())
        .def(py::init<std::string_view>())

        // NOLINTNEXTLINE(misc-redundant-expression)
        .def(py::hash(py::self)).def(py::self == py::self).def(py::self != py::self)

        // Allows equality check with other `str`-likes.
        .def("__eq__", [](const istring &lhs, std::string_view rhs) {
            return static_cast<std::string_view>(lhs) == rhs;
        })
        .def("__ne__", [](const istring &lhs, std::string_view rhs) {
            return static_cast<std::string_view>(lhs) != rhs;
        })

        // To be consistent with `str`, we return the UTF-8 code point length
        // instead of the byte length.
        .def("__len__", &istring::get_code_point_length)

        // To be consistent with `str`, we return the string in single-quotes.
        .def("__repr__", [](const istring &self) {
            return fmt::format("'{}'", self);
        })
        .def("to_py", &istring::operator std::string_view,
R"docstr(
    to_py()

    Converts to ``str``.

    :returns:
        A representation of this string.
    :rtype:
        str
)docstr");

}
