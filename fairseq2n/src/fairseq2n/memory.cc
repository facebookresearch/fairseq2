// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/memory.h"

#include <algorithm>

#include <new>

using namespace fairseq2n::detail;

namespace fairseq2n {
namespace detail {
namespace {

void
deallocate(const void *addr, std::size_t, void *) noexcept
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

    return writable_memory_block{static_cast<std::byte *>(addr), size, nullptr, deallocate};
}

writable_memory_block
copy_memory(memory_span source)
{
    writable_memory_block target = allocate_memory(source.size());

    std::copy(source.begin(), source.end(), target.begin());

    return target;
}

}  // namespace fairseq2n
