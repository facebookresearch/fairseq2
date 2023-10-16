// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/module.h"

#include <cstdint>
#include <memory>

#include <ATen/Device.h>
#include <ATen/ScalarType.h>

#include <fairseq2n/float.h>
#include <fairseq2n/data/image/png_decoder.h>

namespace py = pybind11;

namespace fairseq2n {

void
def_png(py::module_ &data_module)
{
    py::module_ m = data_module.def_submodule("png");

    // PNGDecoder
    py::class_<png_decoder, std::shared_ptr<png_decoder>>(m, "PNGDecoder")
        .def(
            py::init([](
                std::optional<at::ScalarType> maybe_dtype,
                std::optional<at::Device> maybe_device,
                bool pin_memory)
            {
                auto opts = png_decoder_options()
                    .maybe_dtype(maybe_dtype).maybe_device(maybe_device).pin_memory(pin_memory);

                return std::make_shared<png_decoder>(opts);
            }),
            py::arg("dtype") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("pin_memory") = false)
        .def("__call__", &png_decoder::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<png_decoder>();
}
}  // namespace fairseq2n
