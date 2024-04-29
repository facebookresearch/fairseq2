// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/module.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#include <c10/util/Logging.h>

namespace py = pybind11;

namespace fairseq2n {

PYBIND11_MODULE(bindings, m)
{
    py::options opts{};
    opts.disable_function_signatures();

    def_data(m);

    def_memory(m);

    m.def(
        "_enable_aten_logging",
        [](const std::filesystem::path &log_file)
        {
            static std::ofstream log_file_stream{};

            if (log_file_stream.is_open())
                return;

            log_file_stream.open(log_file);

            // Redirect stderr to `log_file`.
            std::cerr.rdbuf(log_file_stream.rdbuf());

            c10::ShowLogInfoToStderr();
        },
        py::arg("log_file"));
}

}  // namespace fairseq2n
