// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>

#include <fairseq2/native/config.h>

namespace py = pybind11;

using fairseq2::cuda_version_major;
using fairseq2::cuda_version_minor;

using fairseq2::supports_cuda;

PYBIND11_MODULE(config, m)
{
    py::options opts{};
    opts.disable_function_signatures();

    // clang-format off

// See https://github.com/llvm/llvm-project/issues/57123.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunreachable-code-return"

    m.def("cuda_version", [] {
        if constexpr (cuda_version_major)
            return py::make_tuple(*cuda_version_major, *cuda_version_minor);
        else
            return py::none();
    },
R"docstr(
    cuda_version()

    Returns the version of CUDA with which the library was built.

    :returns:
        The major and minor version segments.
    :rtype:
        Optional[Tuple[int, int]]
)docstr");

    m.def("supports_cuda", [] {
        return supports_cuda;
    },
R"docstr(
    supports_cuda()

    Indicates whether the library supports CUDA.

    :returns:
        A boolean value indicating whether the library supports CUDA.
    :rtype:
        bool
)docstr");

#pragma clang diagnostic pop

    // clang-format on
}
