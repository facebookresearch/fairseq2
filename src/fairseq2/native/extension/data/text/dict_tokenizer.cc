// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fairseq2/native/data/text/dict_tokenizer/dict_encoder.h"
#include "fairseq2/native/data/text/dict_tokenizer/dict_decoder.h"

namespace py = pybind11;

namespace fairseq2 {

void
def_dict_tokenizer(py::module_ &text_module)
{
    py::module_ m = text_module.def_submodule("dict_tokenizer");

    // DictModel
    py::class_<dict_model>(m, "DictModel")
        .def(
            py::init([](std::vector<std::string> vocab)
            {
                return std::make_unique<dict_model>(std::move(vocab));
            }),
            py::arg("vocab"))

        .def(
            py::pickle(
                [](const dict_model &self)
                {
                    return self.vocab();
                },
                [](std::vector<std::string> vocab)
                {
                    return std::make_unique<dict_model>(std::move(vocab), false);
                }))

        .def("token_to_index", &dict_model::token_to_index)
        .def("index_to_token", &dict_model::index_to_token)

        .def_property_readonly(
            "unk_idx",
            [](const dict_model &self)
            {
                return self.unk_token_idx;
            })
        .def_property_readonly(
            "bos_idx",
            [](const dict_model &self)
            {
                return self.bos_token_idx;
            })
        .def_property_readonly(
            "eos_idx",
            [](const dict_model &self)
            {
                return self.eos_token_idx;
            })
        .def_property_readonly(
            "pad_idx",
            [](const dict_model &self)
            {
                return self.pad_token_idx;
            })

        .def_property_readonly("vocab_size", &dict_model::vocab_size);

    // DictEncoder
    py::class_<dict_encoder, std::shared_ptr<dict_encoder>>(m, "DictEncoder")
        .def(
            py::init([](const dict_model *model, std::int64_t max_seq_len)
            {
                return dict_encoder(model, max_seq_len);
            }),
            py::keep_alive<1, 2>{},
            py::arg("vocab"),
            py::arg("max_seq_len"))
        .def("__call__", &dict_encoder::operator(), py::call_guard<py::gil_scoped_release>{});

    // DictDecoder
    py::class_<dict_decoder, std::shared_ptr<dict_decoder>>(m, "DictDecoder")
        .def(
            py::init([](const dict_model *model)
            {
                return dict_decoder(model);
            }),
            py::keep_alive<1, 2>{},
            py::arg("vocab"))
        .def("__call__", &dict_decoder::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<dict_encoder>();
    map_functors().register_<dict_decoder>();
}

}  // namespace fairseq2
