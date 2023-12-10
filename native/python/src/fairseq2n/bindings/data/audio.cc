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
#include <fairseq2n/data/audio/audio_decoder.h>
#include <fairseq2n/data/audio/waveform_to_fbank_converter.h>

namespace py = pybind11;

namespace fairseq2n {

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

                return std::make_shared<audio_decoder>(opts);
            }),
            py::arg("dtype") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("pin_memory") = false)
        .def("__call__", &audio_decoder::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<audio_decoder>();

    // WaveformToFbankConverter
    py::class_<waveform_to_fbank_converter, std::shared_ptr<waveform_to_fbank_converter>>(
        m, "WaveformToFbankConverter")
        .def(
            py::init([](
                std::int32_t num_mel_bins,
                float32 waveform_scale,
                bool channel_last,
                bool standardize,
                bool keep_waveform,
                std::optional<at::ScalarType> dtype,
                std::optional<at::Device> device,
                bool pin_memory)
            {
                return std::make_shared<waveform_to_fbank_converter>(
                    fbank_options()
                        .num_mel_bins(num_mel_bins)
                        .waveform_scale(waveform_scale)
                        .channel_last(channel_last)
                        .standardize(standardize)
                        .keep_waveform(keep_waveform)
                        .maybe_dtype(dtype)
                        .maybe_device(device)
                        .pin_memory(pin_memory));
            }),
            py::arg("num_mel_bins") = 80,
            py::arg("waveform_scale") = 1.0,
            py::arg("channel_last") = false,
            py::arg("standardize") = false,
            py::arg("keep_waveform") = false,
            py::arg("dtype") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("pin_memory") = false)
        .def("__call__", &waveform_to_fbank_converter::operator());

    map_functors().register_<waveform_to_fbank_converter>();
}

}  // namespace fairseq2n
