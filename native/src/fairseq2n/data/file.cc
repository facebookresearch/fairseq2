// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/file.h"

#include <string_view>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>

#include "fairseq2n/data/byte_stream.h"
#include "fairseq2n/data/file_stream.h"
#include "fairseq2n/data/memory_stream.h"
#include "fairseq2n/data/detail/file.h"
#include "fairseq2n/data/text/utf8_stream.h"
#include "fairseq2n/detail/error.h"
#include "fairseq2n/detail/exception.h"

using namespace fairseq2n::detail;

namespace fairseq2n {
namespace detail {
namespace {

file_desc
do_open_file(const std::filesystem::path &path)
{
    file_desc fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd != invalid_fd)
        return fd;

    std::error_code err = last_error();

    if (err == std::errc::no_such_file_or_directory)
        throw_<byte_stream_error>("'{}' does not exist.", path.string());

    if (err == std::errc::permission_denied) {
        throw_<byte_stream_error>("The permission to read '{}' has been denied.", path.string());
    }

    throw_system_error(err,
        "'{}' cannot be opened", path.string());
}

void
hint_sequential_memory(const memory_block &block, const std::filesystem::path &) noexcept
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
open_file(const std::filesystem::path &path, const file_options &opts)
{
    file_desc fd = do_open_file(path);

    std::size_t chunk_size = opts.maybe_block_size().value_or(0x0010'0000);  // 1 MiB

    std::unique_ptr<byte_stream> stream{};

    if (opts.memory_map()) {
        memory_block block = memory_map_file(fd, path);

        hint_sequential_memory(block, path);

        stream = std::make_unique<memory_stream>(std::move(block));
    } else
        stream = std::make_unique<file_stream>(std::move(fd), path, chunk_size);

// TODO: fix!
//    if (opts.mode() == file_mode::text)
//        stream = std::make_unique<utf8_stream>(
//            std::move(stream), opts.maybe_text_encoding(), chunk_size);

    return stream;
}

memory_block
memory_map_file(const std::filesystem::path &path, bool hint_sequential)
{
    file_desc fd = do_open_file(path);

    memory_block block = memory_map_file(fd, path);

    if (hint_sequential)
        hint_sequential_memory(block, path);

    return block;
}

}
