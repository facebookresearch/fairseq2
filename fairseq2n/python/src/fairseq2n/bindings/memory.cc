// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/bindings/module.h"

#include <cstddef>
#include <utility>

#include <fairseq2n/memory.h>
#include <fairseq2n/detail/exception.h>
#include <fairseq2n/utils/cast.h>

namespace py = pybind11;

using namespace fairseq2n::detail;

namespace fairseq2n {
namespace detail {
namespace {

std::size_t
compute_buffer_size(const py::buffer_info &info)
{
    py::ssize_t size = info.itemsize;

    for (std::size_t i = info.shape.size(); i > 0; --i) {
        if (info.strides[i - 1] != size)
            throw_<std::invalid_argument>(
                "The specified buffer must be contiguous to be used as a `MemoryBlock`.");

        size *= info.shape[i - 1];
    }

    return static_cast<std::size_t>(size);
}

void
release_py_buffer(const void *, std::size_t, void *ctx) noexcept  // NOLINT(bugprone-exception-escape)
{
    py::gil_scoped_acquire gil{};

    auto buffer = static_cast<Py_buffer *>(ctx);

    PyBuffer_Release(buffer);

    delete buffer;  // NOLINT(cppcoreguidelines-owning-memory)
}

memory_block
memory_block_from_buffer(const py::buffer &buffer, bool copy)
{
    py::buffer_info info = buffer.request();

    auto data = static_cast<memory_block::const_pointer>(info.ptr);

    std::size_t size = compute_buffer_size(info);

    if (copy)
        return copy_memory({data, size});

    // Steal the raw buffer pointer.
    Py_buffer *ptr = std::exchange(info.view(), nullptr);

    return memory_block{data, size, ptr, release_py_buffer};
}

}  // namespace
}  // namespace detail

void
def_memory(py::module_ &base_module)
{
    py::module_ m = base_module.def_submodule("memory");

    // MemoryBlock
    py::class_<memory_block>(m, "MemoryBlock", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init(&memory_block_from_buffer), py::arg("buffer"), py::arg("copy") = false)

        .def("__len__", &memory_block::size)

        .def_buffer(
            [](const memory_block &self)
            {
                using T = memory_block::value_type;

                return py::buffer_info{
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                    const_cast<T *>(self.data()), sizeof(T), "B", ssize(self), /*readonly=*/true
                };
            })

        .def(
            py::pickle(
                [](const py::object &self)
                {
                    return py::reinterpret_steal<py::object>(
                        ::PyPickleBuffer_FromObject(self.ptr()));
                },
                [](const py::object &bytes) -> memory_block
                {
                    auto buffer = bytes.cast<py::buffer>();

                    return memory_block_from_buffer(buffer, /*copy=*/false);
                }));
}

}  // namespace fairseq2n
