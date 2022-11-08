// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/utils/memory.h"

#include <cstdlib>
#include <new>

namespace fairseq2::detail {
namespace {

void
heap_deallocate(const void *ptr, std::size_t) noexcept
{
    if (ptr != nullptr)
        ::free(const_cast<void *>(ptr));  // NOLINT
}

}  // namespace

mutable_memory_block
allocate_host_memory(std::size_t size)
{
    void *ptr = ::malloc(size);  // NOLINT
    if (ptr == nullptr)
        throw std::bad_alloc{};

    auto *data = static_cast<std::byte *>(ptr);

    return mutable_memory_block{data, size, heap_deallocate};
}

}  // namespace fairseq2::detail
