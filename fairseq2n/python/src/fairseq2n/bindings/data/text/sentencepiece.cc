// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/module.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fairseq2n/data/text/sentencepiece/sentencepiece.h>

namespace py = pybind11;

namespace fairseq2n {

void
def_sentencepiece(py::module_ &text_module)
{
    py::module_ m = text_module.def_submodule("sentencepiece");

    py::class_<sp_model, std::shared_ptr<sp_model>>(m, "SentencePieceModel")
        .def(
            py::init([](
                std::string_view pathname,
                std::optional<std::vector<std::string>> maybe_control_symbols)
            {
                sp_model_options opts{};

                if (maybe_control_symbols)
                    opts.control_symbols() = *std::move(maybe_control_symbols);

                return std::make_shared<sp_model>(pathname, std::move(opts));
            }),
            py::arg("pathname"),
            py::arg("control_symbols") = std::nullopt)

        .def(
            py::pickle(
                [](const sp_model &self)
                {
                    std::string serialized = self.serialize();

                    return py::bytes(serialized);
                },
                [](const py::bytes &bytes)
                {
                    auto serialized = bytes.cast<std::string>();

                    return sp_model::from_serialized(serialized);
                }))

        .def("token_to_index", &sp_model::token_to_index)
        .def("index_to_token", &sp_model::index_to_token)

        .def_property_readonly("unk_idx", &sp_model::unk_idx)
        .def_property_readonly("bos_idx", &sp_model::bos_idx)
        .def_property_readonly("eos_idx", &sp_model::eos_idx)
        .def_property_readonly("pad_idx", &sp_model::pad_idx)

        .def_property_readonly("vocabulary_size", &sp_model::vocabulary_size);

    py::class_<sp_encoder, std::shared_ptr<sp_encoder>>(m, "SentencePieceEncoder")
        .def(
            py::init([](
                std::shared_ptr<const sp_model> model,
                std::optional<std::vector<std::string>> maybe_prefix_tokens,
                std::optional<std::vector<std::string>> maybe_suffix_tokens,
                bool reverse,
                bool enable_sampling,
                std::int32_t nbest_size,
                float alpha,
                std::optional<at::Device> maybe_device,
                bool pin_memory)
            {
                auto opts = sp_encoder_options()
                    .reverse(reverse)
                    .enable_sampling(enable_sampling)
                    .nbest_size(nbest_size)
                    .alpha(alpha)
                    .maybe_device(maybe_device)
                    .pin_memory(pin_memory);

                if (maybe_prefix_tokens)
                    opts.prefix_tokens() = *std::move(maybe_prefix_tokens);

                if (maybe_suffix_tokens)
                    opts.suffix_tokens() = *std::move(maybe_suffix_tokens);

                return std::make_shared<sp_encoder>(std::move(model), std::move(opts));
            }),
            py::arg("model"),
            py::arg("prefix_tokens")   = std::nullopt,
            py::arg("suffix_tokens")   = std::nullopt,
            py::arg("reverse")         = false,
            py::arg("enable_sampling") = false,
            py::arg("nbest_size")      = -1,
            py::arg("alpha")           = 0.1,
            py::arg("device")          = std::nullopt,
            py::arg("pin_memory")      = false)
        .def(
            "__call__",
            &sp_encoder::operator(),
            py::arg("text"),
            py::call_guard<py::gil_scoped_release>{})
        .def(
            "encode_as_tokens",
            &sp_encoder::encode_as_tokens,
            py::arg("text"),
            py::call_guard<py::gil_scoped_release>{})

        .def_property_readonly("prefix_indices", &sp_encoder::prefix_indices)
        .def_property_readonly("suffix_indices", &sp_encoder::suffix_indices);

    py::class_<sp_decoder, std::shared_ptr<sp_decoder>>(m, "SentencePieceDecoder")
        .def(
            py::init<std::shared_ptr<const sp_model>, bool>(),
            py::arg("model"),
            py::arg("reverse") = false)
        .def(
            "__call__",
            &sp_decoder::operator(),
            py::arg("token_indices"),
            py::call_guard<py::gil_scoped_release>{})
        .def(
            "decode_from_tokens",
            &sp_decoder::decode_from_tokens,
            py::arg("tokens"),
            py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<sp_encoder>();
    map_functors().register_<sp_decoder>();
}

}  // namespace fairseq2n
