// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/memory.h"

#include <new>

namespace fairseq2 {
namespace detail {
namespace {

void
deallocate(const void *addr, std::size_t) noexcept
{
    if (addr != nullptr)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        ::operator delete(const_cast<void *>(addr));
}

}  // namespace
}  // namespace detail

writable_memory_block
allocate_memory(std::size_t size)
{
    void *addr = ::operator new(size);

    return writable_memory_block{static_cast<std::byte *>(addr), size, detail::deallocate};
}

}  // namespace fairseq2
