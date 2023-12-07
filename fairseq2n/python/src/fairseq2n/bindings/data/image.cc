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
#include <fairseq2n/data/image/image_decoder.h>

namespace py = pybind11;

namespace fairseq2n {

void
def_image(py::module_ &data_module)
{
    py::module_ m = data_module.def_submodule("image");

    // ImageDecoder
    py::class_<image_decoder, std::shared_ptr<image_decoder>>(m, "ImageDecoder")
        .def(
            py::init([](
                std::optional<at::Device> maybe_device,
                bool pin_memory)
            {
                auto opts = image_decoder_options()
                    .maybe_device(maybe_device).pin_memory(pin_memory);

                return std::make_shared<image_decoder>(opts);
            }),
            py::arg("device") = std::nullopt,
            py::arg("pin_memory") = false)
        .def("__call__", &image_decoder::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<image_decoder>();
}
}  // namespace fairseq2n
