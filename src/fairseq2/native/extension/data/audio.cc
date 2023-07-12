// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/module.h"

#include <memory>

#include <ATen/Device.h>
#include <ATen/ScalarType.h>

#include "fairseq2/native/data/audio/audio_decoder.h"

namespace py = pybind11;

namespace fairseq2 {

void
def_audio(py::module_ &data_module)
{
    py::module_ m = data_module.def_submodule("audio");

    // AudioDecoder
    py::class_<audio_decoder, std::shared_ptr<audio_decoder>>(m, "AudioDecoder")
        .def(
            py::init([](
                std::optional<at::ScalarType> maybe_dtype,
                std::optional<at::Device> maybe_device,
                bool pin_memory)
            {
                auto opts = audio_decoder_options()
                    .maybe_dtype(maybe_dtype).maybe_device(maybe_device).pin_memory(pin_memory);

                return audio_decoder{opts};
            }),
            py::arg("dtype") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("pin_memory") = false)
        .def("__call__", &audio_decoder::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<audio_decoder>();
}

}  // namespace fairseq2
