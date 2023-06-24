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
#include <fairseq2/native/data/processors/list_processor.h>
#include <fairseq2/native/data/processors/str_splitter.h>
#include <fairseq2/native/data/processors/str_to_int_converter.h>
#include <fairseq2/native/data/processors/str_to_tensor_converter.h>

#include "fairseq2/native/extension/data/utils.h"

namespace py = pybind11;

namespace fairseq2 {

void
def_processors(py::module_ &data_module)
{
    py::module_ m = data_module.def_submodule("processors");

    // ListProcessor
    py::class_<list_processor, data_processor, std::shared_ptr<list_processor>>(m, "ListProcessor")
        .def(
            py::init([](
                const py::args &args,
                std::optional<std::vector<std::size_t>> &&indices,
                bool disable_parallelism)
            {
                std::vector<std::shared_ptr<const data_processor>> procs{};

                procs.reserve(args.size());

                for (py::handle h : args)
                    procs.push_back(as_data_processor(h));

                return list_processor(std::move(procs), std::move(indices), disable_parallelism);
            }),
            py::arg("indices") = std::nullopt,
            py::arg("disable_parallelism") = false);

    // StrSplitter
    py::class_<str_splitter, data_processor, std::shared_ptr<str_splitter>>(m, "StrSplitter")
        .def(
            py::init([](std::string_view sep)
            {
                if (sep.size() != 1)
                    throw std::invalid_argument{"`sep` must be of length 1."};

                return str_splitter{sep[0]};
            }),
            py::arg("sep") = '\t');

    // StrToIntConverter
    py::class_<
        str_to_int_converter,
        data_processor,
        std::shared_ptr<str_to_int_converter>>(m, "StrToIntConverter")
        .def(py::init<std::int16_t>(), py::arg("base") = 10);

    // StrToTensorConverter
    py::class_<
        str_to_tensor_converter,
        data_processor,
        std::shared_ptr<str_to_tensor_converter>>(m, "StrToTensorConverter")
        .def(
            py::init<std::optional<std::vector<std::int64_t>>, std::optional<at::ScalarType>>(),
            py::arg("size") = std::nullopt,
            py::arg("dtype") = std::nullopt);
}

}  // namespace fairseq2
