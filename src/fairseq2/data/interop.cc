// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/data/interop.h"

#include <cstdint>
#include <string_view>
#include <stdexcept>

#include <ATen/Tensor.h>
#include <fmt/core.h>
#include <pybind11/operators.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

using fairseq2::istring;
using fairseq2::ivariant;

namespace {

inline bool
is_tensor(const py::handle &h)
{
    return THPVariable_Check(h.ptr());
}

}  // namespace

py::object
py::detail::type_caster<ivariant>::cast_from_cc(const ivariant &src)
{
    if (src.is_none())
        return none();

    if (src.is_bool())
        return py::cast(src.as_bool());

    if (src.is_int())
        return py::cast(src.as_int());

    if (src.is_double())
        return py::cast(src.as_double());

    if (src.is_string())
        return py::cast(src.as_string());

    if (src.is_tensor())
        return py::cast(src.as_tensor());

    throw std::runtime_error{"The value cannot be converted to a Python object."};
}

ivariant
py::detail::type_caster<ivariant>::cast_from_py(py::handle src)
{
    if (src.is_none())
        return ivariant::none();

    if (py::isinstance<py::bool_>(src))
        return src.cast<bool>();

    if (py::isinstance<py::int_>(src))
        return src.cast<std::int64_t>();

    if (py::isinstance<py::float_>(src))
        return src.cast<double>();

    if (py::isinstance<py::str>(src))
        return src.cast<std::string_view>();

    if (py::isinstance<istring>(src))
        return src.cast<istring>();

    if (is_tensor(src))
        return src.cast<at::Tensor>();

    return ivariant{};
}

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
        .def(py::init<std::string_view>(),
R"docstr(
    __init__(s=None)

    :param s:
        The string to copy.

    :type s:
        ~typing.Optional[str]
)docstr")

        // To be consistent with `str`, we return the UTF-8 code point length
        // instead of the byte length.
        .def("__len__", &istring::get_code_point_length)

        .def(py::self == py::self)  // NOLINT(misc-redundant-expression)
        .def(py::self != py::self)  // NOLINT(misc-redundant-expression)

        // Allows equality check with other `str`-likes.
        .def("__eq__", [](const istring &lhs, std::string_view rhs) {
            return static_cast<std::string_view>(lhs) == rhs;
        })
        .def("__ne__", [](const istring &lhs, std::string_view rhs) {
            return static_cast<std::string_view>(lhs) != rhs;
        })

        .def(py::hash(py::self))

        // To be consistent with `str`, we return the string in single-quotes.
        .def("__repr__", [](const istring &self) {
            return fmt::format("'{}'", self);
        })

        .def("to_py", &istring::operator std::string_view,
R"docstr(
    to_py()

    Converts to ``str``.

    :returns:
        A ``str`` representation of this string.
    :rtype:
        str
)docstr");
}
