// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/module.h"

namespace py = pybind11;

namespace fairseq2n {

PYBIND11_MODULE(bindings, m)
{
    py::options opts{};
    opts.disable_function_signatures();

    def_data(m);

    def_memory(m);
}

}  // namespace fairseq2n
