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
#include <fairseq2n/data/video/video_decoder.h>

namespace py = pybind11;

namespace fairseq2n {

void
def_video(py::module_ &data_module)
{
    py::module_ m = data_module.def_submodule("video");

    // VideoDecoder
    py::class_<video_decoder, std::shared_ptr<video_decoder>>(m, "VideoDecoder")
        .def(
            py::init([](
                std::optional<at::ScalarType> maybe_dtype,
                std::optional<at::Device> maybe_device,
                bool pin_memory,
                bool get_pts_only,
                bool get_frames_only,
                int width = 0,
                int height = 0)
            {
                auto opts = video_decoder_options()
                    .maybe_dtype(maybe_dtype)
                    .maybe_device(maybe_device)
                    .pin_memory(pin_memory)
                    .get_pts_only(get_pts_only)
                    .get_frames_only(get_frames_only)
                    .width(width)
                    .height(height);
                
                return std::make_shared<video_decoder>(opts);
            }),
            py::arg("dtype") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("pin_memory") = false,
            py::arg("get_pts_only") = false,
            py::arg("get_frames_only") = false,
            py::arg("width") = 0,
            py::arg("height") = 0)
        .def("__call__", &video_decoder::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<video_decoder>();
}
}  // namespace fairseq2n
