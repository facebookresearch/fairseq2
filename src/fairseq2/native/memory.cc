// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/memory.h"

#include <algorithm>

#include <new>

namespace fairseq2 {
namespace detail {
namespace {

void
deallocate(const void *addr, std::size_t, void *) noexcept
{
    if (addr != nullptr)
        ::operator delete(const_cast<void *>(addr)); // NOLINT(cppcoreguidelines-pro-type-const-cast)
}

}  // namespace
}  // namespace detail

writable_memory_block
allocate_memory(std::size_t size)
{
    void *addr = ::operator new(size);

    return writable_memory_block{static_cast<std::byte *>(addr), size, nullptr, detail::deallocate};
}

writable_memory_block
copy_memory(memory_span src)
{
    writable_memory_block blk = allocate_memory(src.size());

    std::copy(src.begin(), src.end(), blk.begin());

    return blk;
}

}  // namespace fairseq2
