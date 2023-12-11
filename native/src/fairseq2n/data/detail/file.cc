// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/detail/file.h"

#include <system_error>

#include <sys/mman.h>
#include <sys/stat.h>

#include "fairseq2n/memory.h"
#include "fairseq2n/detail/error.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {
namespace {

void
mmap_deallocate(const void *addr, std::size_t size, void *) noexcept
{
    if (addr != nullptr)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        ::munmap(const_cast<void *>(addr), size);
}

}  // namespace

memory_block
memory_map_file(const file_desc &fd, std::string_view pathname)
{
    struct ::stat buf{};
    if (::fstat(fd.get(), &buf) == -1)
        throw_system_error(last_error(),
            "The file size of '{}' cannot be determined", pathname);

    auto size = static_cast<std::size_t>(buf.st_size);
    if (size == 0)
        return memory_block{};

    void *addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd.get(), 0);
    if (addr == MAP_FAILED)
        throw_system_error(last_error(),
            "'{}' cannot be memory mapped", pathname);

    return memory_block{static_cast<std::byte *>(addr), size, nullptr, mmap_deallocate};
}

}  // namespace fairseq2n::detail
