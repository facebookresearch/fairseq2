// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/data/utils.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

#include <fairseq2/native/py.h>
#include <fairseq2/native/data/data_processor.h>
#include <fairseq2/native/data/processors/str_to_int_converter.h>
#include <fairseq2/native/data/processors/str_to_tensor_converter.h>

#include "fairseq2/native/extension/module.h"

namespace py = pybind11;

using namespace fairseq2::detail;

namespace fairseq2 {
namespace detail {
namespace {

using data_process_fn = std::function<data(const data &)>;

class custom_data_processor : public data_processor {
public:
    explicit
    custom_data_processor(data_process_fn &&fn) noexcept
      : fn_{std::move(fn)}
    {}

    data
    operator()(const data &d) const override
    {
        return fn_(d);
    }

    data
    operator()(data &&d) const override
    {
        return (*this)(d);
    }

private:
    data_process_fn fn_;
};

}  // namespace
}  // namespace detail

std::shared_ptr<data_processor>
as_data_processor(py::handle h)
{
    // DataProcessor
    if (py::isinstance<data_processor>(h))
        return h.cast<std::shared_ptr<data_processor>>();

    static py::module_ builtins = py::module_::import("builtins");

    // Int
    if (h.is(builtins.attr("int")))
        return std::make_shared<str_to_int_converter>();

    static py::module_ torch = py::module_::import("torch");

    // Tensor
    if (h.is(torch.attr("tensor")))
        return std::make_shared<str_to_tensor_converter>();

    // Callable
    if (py::isinstance<py::function>(h))
        return std::make_shared<custom_data_processor>(h.cast<data_process_fn>());

    throw std::invalid_argument{"The specified object must be callable."};
}

}  // namespace fairseq2
