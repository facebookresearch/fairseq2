// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/module.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fairseq2/native/data/data_pipeline.h>
#include <fairseq2/native/data/data_processor.h>
#include <fairseq2/native/data/text/sentencepiece/sentencepiece.h>
#include <fairseq2/native/data/text/text.h>

namespace py = pybind11;

namespace fairseq2 {
namespace {

void
def_sentencepiece(py::module_ &base)
{
    py::module_ m = base.def_submodule("sentencepiece");

    py::class_<sp_model>(m, "SentencePieceModel")
        .def(py::init(
            [](std::string_view pathname,
               std::optional<std::vector<std::string>> &&control_tokens,
               bool add_bos,
               bool add_eos,
               bool reverse)
            {
                auto opts = sp_model_options()
                    .add_bos(add_bos)
                    .add_eos(add_eos)
                    .reverse(reverse);

                if (control_tokens)
                    opts.control_tokens() = *std::move(control_tokens);

                return std::make_unique<sp_model>(pathname, opts);
            }),
            py::arg("pathname"),
            py::arg("control_tokens") = std::nullopt,
            py::arg("add_bos")        = false,
            py::arg("add_eos")        = false,
            py::arg("reverse")        = false)
        .def("token_to_index", &sp_model::token_to_index)
        .def("index_to_token", &sp_model::index_to_token)
        .def_property_readonly("unk_idx", &sp_model::unk_idx)
        .def_property_readonly("bos_idx", &sp_model::bos_idx)
        .def_property_readonly("eos_idx", &sp_model::eos_idx)
        .def_property_readonly("pad_idx", &sp_model::pad_idx)
        .def_property_readonly("vocabulary_size", &sp_model::vocabulary_size);

    py::class_<sp_encoder, data_processor>(m, "SentencePieceEncoder")
        .def(py::init(
            [](const sp_model *model,
               bool enable_sampling,
               std::int32_t nbest_size,
               float alpha,
               std::optional<std::int64_t> batch_size,
               std::optional<std::int64_t> pad_to_length,
               std::int64_t pad_to_multiple,
               bool left_pad,
               at::ScalarType dtype,
               std::optional<at::Device> device,
               bool pin_memory,
               bool disable_parallelism)
            {
                auto opts = sp_encoder_options()
                    .enable_sampling(enable_sampling)
                    .nbest_size(nbest_size)
                    .alpha(alpha)
                    .batch_size(batch_size)
                    .pad_to_length(pad_to_length)
                    .pad_to_multiple(pad_to_multiple)
                    .left_pad(left_pad)
                    .dtype(dtype)
                    .device(device)
                    .pin_memory(pin_memory)
                    .disable_parallelism(disable_parallelism);

                return sp_encoder{model, opts};
            }),
            py::keep_alive<1, 2>{},
            py::arg("model"),
            py::arg("enable_sampling")     = false,
            py::arg("nbest_size")          = -1,
            py::arg("alpha")               = 0.1,
            py::arg("batch_size")          = std::nullopt,
            py::arg("pad_to_length")       = std::nullopt,
            py::arg("pad_to_multiple")     = 1,
            py::arg("left_pad")            = false,
            py::arg("dtype")               = at::kInt,
            py::arg("device")              = std::nullopt,
            py::arg("pin_memory")          = false,
            py::arg("disable_parallelism") = false);

    py::class_<sp_decoder, data_processor>(m, "SentencePieceDecoder")
        .def(py::init<const sp_model *>(), py::keep_alive<1, 2>{}, py::arg("model"));
}

}  // namespace

void
def_text(py::module_ &base)
{
    py::module_ m = base.def_submodule("text");

    py::enum_<line_ending>(m, "LineEnding")
        .value("INFER", line_ending::infer)
        .value("LF",    line_ending::lf)
        .value("CRLF",  line_ending::crlf);

    m.def("read_text",
        [](std::string &&pathname,
           std::string &&encoding,
           line_ending le,
           bool ltrim,
           bool rtrim,
           bool skip_empty,
           bool memory_map,
           std::optional<std::size_t> block_size)
        {
            auto opts = text_options()
                .encoding(std::move(encoding))
                .line_ending(le)
                .ltrim(ltrim)
                .rtrim(rtrim)
                .skip_empty(skip_empty)
                .memory_map(memory_map)
                .block_size(block_size);

            return read_text(std::move(pathname), std::move(opts));
        },
        py::arg("pathname"),
        py::arg("encoding")    = "",
        py::arg("line_ending") = line_ending::infer,
        py::arg("ltrim")       = false,
        py::arg("rtrim")       = false,
        py::arg("skip_empty")  = false,
        py::arg("memory_map")  = false,
        py::arg("block_size")  = std::nullopt);

    def_sentencepiece(m);
}

}  // namespace fairseq2
