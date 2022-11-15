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
#include <fmt/format.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

using fairseq2::idict;
using fairseq2::ilist;
using fairseq2::istring;
using fairseq2::ivariant;

template <>
inline bool
py::isinstance<at::Tensor>(handle obj)
{
    return THPVariable_Check(obj.ptr());
}

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

    if (src.is_list())
        return py::cast(src.as_shared_list());

    if (src.is_dict())
        return py::cast(src.as_shared_dict());

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

    if (py::isinstance<at::Tensor>(src))
        return src.cast<at::Tensor>();

    if (py::isinstance<ilist>(src))
        return src.cast<std::shared_ptr<ilist>>();

    if (py::isinstance<idict>(src))
        return src.cast<std::shared_ptr<idict>>();

    if (py::isinstance<py::iterable>(src)) {
        auto lst = std::make_shared<ilist>();

        for (py::handle h : src)
            lst->push_back(h.cast<ivariant>());

        return lst;
    }

    return ivariant{};
}

namespace {

void register_to_abcs()
{
    auto col_abc = py::module_::import("collections.abc");

    auto ms = col_abc.attr("MutableSequence");
    auto mm = col_abc.attr("MutableMapping");

    ms.attr("register")(py::type::of<ilist>());
    mm.attr("register")(py::type::of<idict>());
}

}  // namespace

PYBIND11_MODULE(interop, m)
{
    py::options opts{};
    opts.disable_function_signatures();

    // istring
    py::class_<istring>(m, "IString",
R"docstr(
    Represents an immutable UTF-8 string that supports zero-copy marshalling
    between Python and native code.
)docstr")
        .def(py::init<>())
        .def(py::init<std::string_view>(),
R"docstr(
    __init__(s=None)

    :param s:
        The Python string to copy.

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

    // ilist
    py::bind_vector<ilist, std::shared_ptr<ilist>>(m, "IList",
R"docstr(
    Holds :data:`~fairseq2.data.IVariant` elements that can be zero-copy
    marshalled between Python and native code.
)docstr")
        .def("__repr__", [](const py::iterable &self) {
            return fmt::format("[{}]", fmt::join(self, ", "));
        });

    // idict
    py::bind_map<idict, std::shared_ptr<idict>>(m, "IDict",
R"docstr(
    Holds :data:`~fairseq2.data.IVariant` key/value pairs that can be zero-copy
    marshalled between Python and native code.
)docstr")
        .def("__repr__", [](const py::object &self) {
            py::iterable items = self.attr("items")();

            return fmt::format("{{{:d}}}", fmt::join(items, ", "));
        });

    register_to_abcs();
}

template <>
struct fmt::formatter<py::handle> {
    auto
    parse(fmt::format_parse_context &ctx) noexcept
    {
        auto pos = ctx.begin();

        if (pos != ctx.end() && *pos == 'd') {
            dict_item_ = true;

            pos++;
        }

        return pos;
    }

    template <typename FormatContext>
    auto
    format(const py::handle &h, FormatContext &ctx) const
    {
        if (dict_item_) {
            auto t = h.cast<py::tuple>();

            auto k = static_cast<std::string>(py::repr(t[0]));
            auto v = static_cast<std::string>(py::repr(t[1]));

            return fmt::format_to(ctx.out(),
                "{}: {}", k, v);
        } else {
            auto v = static_cast<std::string>(py::repr(h));

            return fmt::format_to(ctx.out(),
                "{}", v);
        }
    }

private:
    bool dict_item_{};
};
