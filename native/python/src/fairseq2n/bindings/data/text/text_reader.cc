// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/module.h"

#include <cstddef>
#include <optional>
#include <string>

#include <fairseq2n/data/data_pipeline.h>
#include <fairseq2n/data/text/text_reader.h>

namespace py = pybind11;

namespace fairseq2n {

void
def_text_reader(py::module_ &text_module)
{
    py::module_ m = text_module.def_submodule("text_reader");

    py::enum_<line_ending>(m, "LineEnding")
        .value("INFER", line_ending::infer)
        .value("LF",    line_ending::lf)
        .value("CRLF",  line_ending::crlf);

    m.def(
        "read_text",
        [](
            std::string pathname,
            std::optional<std::string> maybe_encoding,
            line_ending le,
            bool ltrim,
            bool rtrim,
            bool skip_empty,
            bool memory_map,
            std::optional<std::size_t> maybe_block_size)
        {
            auto opts = text_options()
                .maybe_encoding(std::move(maybe_encoding))
                .line_ending(le)
                .ltrim(ltrim)
                .rtrim(rtrim)
                .skip_empty(skip_empty)
                .memory_map(memory_map)
                .maybe_block_size(maybe_block_size);

            return read_text(std::move(pathname), std::move(opts));
        },
        py::arg("pathname"),
        py::arg("encoding")    = std::nullopt,
        py::arg("line_ending") = line_ending::infer,
        py::arg("ltrim")       = false,
        py::arg("rtrim")       = false,
        py::arg("skip_empty")  = false,
        py::arg("memory_map")  = false,
        py::arg("block_size")  = std::nullopt);
}

}  // namespace fairseq2n
