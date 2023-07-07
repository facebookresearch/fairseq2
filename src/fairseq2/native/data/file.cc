// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/file.h"

#include <string_view>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>

#include <fmt/core.h>

#include "fairseq2/native/error.h"
#include "fairseq2/native/data/byte_stream.h"
#include "fairseq2/native/data/file_stream.h"
#include "fairseq2/native/data/memory_stream.h"
#include "fairseq2/native/data/detail/file.h"
#include "fairseq2/native/data/text/utf8_stream.h"

using namespace fairseq2::detail;

namespace fairseq2 {
namespace detail {
namespace {

file_desc
do_open_file(const std::string &pathname)
{
    file_desc fd = ::open(pathname.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd != invalid_fd)
        return fd;

    std::error_code err = last_error();

    if (err == std::errc::no_such_file_or_directory)
        throw byte_stream_error{
            fmt::format("'{}' does not exist.", pathname)};

    if (err == std::errc::permission_denied) {
        throw byte_stream_error{
            fmt::format("The permission to read '{}' has been denied.", pathname)};
    }

    throw std::system_error{err,
        fmt::format("'{}' cannot be opened", pathname)};
}

void
hint_sequential_memory(const memory_block &block, std::string_view) noexcept
{
#ifdef __linux__
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto addr = const_cast<std::byte *>(block.data());

    int result = ::madvise(addr, block.size(), MADV_SEQUENTIAL);
    if (result != 0) {
        // TODO: warn
    }
#else
    (void) block;
#endif
}

}  // namespace
}  // namespace detail

std::unique_ptr<byte_stream>
open_file(const std::string &pathname, const file_options &opts)
{
    file_desc fd = do_open_file(pathname);

    std::size_t chunk_size = opts.block_size().value_or(0x0010'0000);  // 1 MiB

    std::unique_ptr<byte_stream> stream{};

    if (opts.memory_map()) {
        memory_block block = memory_map_file(fd, pathname);

        hint_sequential_memory(block, pathname);

        stream = std::make_unique<memory_stream>(std::move(block));
    } else
        stream = std::make_unique<file_stream>(std::move(fd), pathname, chunk_size);

    if (opts.mode() == file_mode::text)
        stream = std::make_unique<utf8_stream>(std::move(stream), opts.text_encoding(), chunk_size);

    return stream;
}

memory_block
memory_map_file(const std::string &pathname, bool sequential)
{
    file_desc fd = do_open_file(pathname);

    memory_block block = memory_map_file(fd, pathname);

    if (sequential)
        hint_sequential_memory(block, pathname);

    return block;
}

}
