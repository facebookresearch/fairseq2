// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/module.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <ATen/ScalarType.h>

#include <fairseq2/native/data/data_pipeline.h>
#include <fairseq2/native/data/data_processor.h>
#include <fairseq2/native/data/processors/string_splitter.h>
#include <fairseq2/native/data/processors/string_to_integer_converter.h>
#include <fairseq2/native/data/processors/string_to_tensor_converter.h>

namespace py = pybind11;

namespace fairseq2 {

void
def_processors(py::module_ &data_module)
{
    py::module_ m = data_module.def_submodule("processors");

    // StrSplitter
    py::class_<string_splitter, data_processor, std::shared_ptr<string_splitter>>(m, "StrSplitter")
        .def(
            py::init([](std::string_view sep, std::optional<std::vector<std::string>> names)
            {
                if (sep.size() != 1)
                    throw std::invalid_argument{"`sep` must be of length 1."};

                return string_splitter{sep[0], std::move(names)};
            }),
            py::arg("sep") = '\t',
            py::arg("names") = std::nullopt);

    // StrToIntConverter
    py::class_<
        string_to_integer_converter,
        data_processor,
        std::shared_ptr<string_to_integer_converter>>(m, "StrToIntConverter")
        .def(py::init<std::int16_t>(), py::arg("base") = 10);

    // StrToTensorConverter
    py::class_<
        string_to_tensor_converter,
        data_processor,
        std::shared_ptr<string_to_tensor_converter>>(m, "StrToTensorConverter")
        .def(
            py::init<std::optional<std::vector<std::int64_t>>, std::optional<at::ScalarType>>(),
            py::arg("size") = std::nullopt,
            py::arg("dtype") = std::nullopt);
}

}  // namespace fairseq2
