// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/module.h"

#include <cstddef>
#include <utility>

#include <fairseq2/native/memory.h>
#include <fairseq2/native/utils/cast.h>

namespace py = pybind11;

using namespace fairseq2::detail;

namespace fairseq2 {
namespace detail {
namespace {

std::size_t
compute_buffer_size(const py::buffer_info &info)
{
    py::ssize_t size = info.itemsize;

    for (std::size_t i = info.shape.size(); i > 0; --i) {
        if (info.strides[i - 1] != size)
            throw std::invalid_argument{"The specified buffer must be contiguous."};

        size *= info.shape[i - 1];
    }

    return static_cast<std::size_t>(size);
}

void
release_buffer(const void *, std::size_t, void *ctx) noexcept  // NOLINT(bugprone-exception-escape)
{
    py::gil_scoped_acquire gil{};

    PyBuffer_Release(static_cast<Py_buffer *>(ctx));
}

}  // namespace
}  // namespace detail

void
def_memory(py::module_ &data_module)
{
    py::module_ m = data_module.def_submodule("memory");

    // MemoryBlock
    py::class_<memory_block>(m, "MemoryBlock", py::buffer_protocol())
        .def(py::init<>())
        .def(
            py::init([](const py::buffer &buf, bool copy) -> memory_block
            {
                py::buffer_info info = buf.request();

                auto data = static_cast<memory_block::const_pointer>(info.ptr);

                std::size_t size = compute_buffer_size(info);

                if (copy)
                    return copy_memory({data, size});

                Py_buffer *ptr = std::exchange(info.view(), nullptr);

                return memory_block{data, size, ptr, release_buffer};
            }),
            py::arg("buffer"),
            py::arg("copy") = false)
        .def_buffer(
            [](const memory_block &self)
            {
                using T = memory_block::value_type;

                return py::buffer_info{
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                    const_cast<T *>(self.data()), sizeof(T), "B", ssize(self), /*readonly=*/true
                };
            });
}

}  // namespace fairseq2
