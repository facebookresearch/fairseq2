// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/module.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <ATen/ScalarType.h>

#include <fairseq2n/data/data_pipeline.h>
#include <fairseq2n/data/text/string_splitter.h>
#include <fairseq2n/data/text/string_to_int_converter.h>
#include <fairseq2n/data/text/string_to_tensor_converter.h>
#include <fairseq2n/detail/exception.h>

namespace py = pybind11;

using namespace fairseq2n::detail;

namespace fairseq2n {

void
def_text_converters(py::module_ &text_module)
{
    py::module_ m = text_module.def_submodule("converters");

    // StrSplitter
    py::class_<string_splitter, std::shared_ptr<string_splitter>>(m, "StrSplitter")
        .def(
            py::init([](
                std::string_view sep,
                std::optional<std::vector<std::string>> maybe_names,
                std::optional<std::vector<std::size_t>> maybe_indices,
                bool exclude)
            {
                if (sep.size() != 1)
                    throw_<std::invalid_argument>(
                        "`sep` must be of length 1, but is of length {} instead.", sep.size());

                std::vector<std::string> names{};
                if (maybe_names)
                    names = *std::move(maybe_names);

                std::vector<std::size_t> indices{};
                if (maybe_indices)
                    indices = *std::move(maybe_indices);

                return std::make_shared<string_splitter>(
                    sep[0], std::move(names), std::move(indices), exclude);
            }),
            py::arg("sep") = '\t',
            py::arg("names") = std::nullopt,
            py::arg("indices") = std::nullopt,
            py::arg("exclude") = false)
        .def("__call__", &string_splitter::operator(), py::call_guard<py::gil_scoped_release>{});

    // StrToIntConverter
    py::class_<string_to_int_converter, std::shared_ptr<string_to_int_converter>>(
        m, "StrToIntConverter")
        .def(py::init<std::int16_t>(), py::arg("base") = 10)
        .def("__call__", &string_to_int_converter::operator());

    // StrToTensorConverter
    py::class_<string_to_tensor_converter, std::shared_ptr<string_to_tensor_converter>>(
        m, "StrToTensorConverter")
        .def(
            py::init([](
                std::optional<std::vector<std::int64_t>> maybe_size,
                std::optional<at::ScalarType> maybe_dtype)
            {
                std::vector<std::int64_t> size{};
                if (maybe_size)
                    size = *std::move(maybe_size);

                return std::make_shared<string_to_tensor_converter>(std::move(size), maybe_dtype);
            }),
            py::arg("size") = std::nullopt,
            py::arg("dtype") = std::nullopt)
        .def(
            "__call__",
            &string_to_tensor_converter::operator(),
            py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<string_splitter>();
    map_functors().register_<string_to_int_converter>();
    map_functors().register_<string_to_tensor_converter>();
}

}  // namespace fairseq2n
