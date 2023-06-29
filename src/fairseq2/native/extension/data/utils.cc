// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/data/utils.h"

#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fairseq2/native/data/data_processor.h>
#include <fairseq2/native/data/processors/composite_data_processor.h>
#include <fairseq2/native/data/processors/custom_data_processor.h>
#include <fairseq2/native/data/processors/element_processor.h>
#include <fairseq2/native/data/processors/string_to_integer_converter.h>
#include <fairseq2/native/data/processors/string_to_tensor_converter.h>

#include "fairseq2/native/extension/module.h"

namespace py = pybind11;

namespace fairseq2::detail {
namespace {

std::shared_ptr<const data_processor>
as_data_processor_core(const py::object &fn)
{
    // DataProcessor
    if (py::isinstance<data_processor>(fn))
        return fn.cast<std::shared_ptr<const data_processor>>();

    static py::module_ builtins = py::module_::import("builtins");

    // int
    if (fn.is(builtins.attr("int")))
        return std::make_shared<string_to_integer_converter>();

    static py::module_ torch = py::module_::import("torch");

    // Tensor
    if (fn.is(torch.attr("tensor")))
        return std::make_shared<string_to_tensor_converter>();

    // Callable
    if (py::isinstance<py::function>(fn))
        return std::make_shared<custom_data_processor>(fn.cast<std::function<data(data &&)>>());

    throw std::invalid_argument{"The specified object must be callable."};
}

std::shared_ptr<const data_processor>
as_data_processor(const py::object &fn)
{
    if (py::isinstance<py::sequence>(fn)) {
        std::vector<std::shared_ptr<const data_processor>> p{};

        for (py::object e : fn.cast<py::sequence>())
            p.push_back(as_data_processor_core(e));

        return std::make_shared<composite_data_processor>(std::move(p));
    } else
        return as_data_processor_core(fn);
}

}  // namespace

std::shared_ptr<const data_processor>
as_data_processor(const py::object &fn, std::optional<std::string_view> selector)
{
    std::shared_ptr<const data_processor> p = as_data_processor(fn);

    if (selector)
        p = std::make_shared<element_processor>(std::move(p), *selector);

    return p;
}

}  // namespace fairseq2::detail
