// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/file_stream.h"

#include <utility>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

#include "fairseq2n/detail/error.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

file_stream::file_stream(file_desc &&fd, std::filesystem::path path, std::size_t chunk_size) noexcept
  : fd_{std::move(fd)}, path_{std::move(path)}, chunk_size_{chunk_size}
{
    hint_sequential_file();
}

void
file_stream::hint_sequential_file() noexcept
{
#ifdef __linux__
    int result = ::posix_fadvise(fd_.get(), 0, 0, POSIX_FADV_SEQUENTIAL);
    if (result != 0) {
        // TODO: warn
    }
#endif
}

memory_block
file_stream::read_chunk()
{
    if (is_eod_)
        return {};

    writable_memory_block chunk = allocate_memory(chunk_size_);

    writable_memory_span remaining_space = chunk;

    while (!remaining_space.empty()) {
        std::size_t num_bytes_read = fill_chunk(remaining_space);
        if (num_bytes_read == 0) {
            is_eod_ = true;

            break;
        }

        remaining_space = remaining_space.subspan(num_bytes_read);
    }

    if (remaining_space.empty())
        return chunk;

    return chunk.share_first(chunk.size() - remaining_space.size());
}

void
file_stream::seek(std::size_t offset)
{
    seek(offset, /*from_current=*/false);
}

std::size_t
file_stream::position() const
{
    return seek(0, /*from_current=*/true);
}

void
file_stream::reset()
{
    seek(0);

    is_eod_ = false;
}

void
file_stream::record_position(tape &t) const
{
    std::size_t offset = position();

    t.record(offset);
}

void
file_stream::reload_position(tape &t)
{
    seek(t.read<std::size_t>());

    is_eod_ = false;
}

bool
file_stream::supports_seek() const noexcept
{
    return true;
}

std::size_t
file_stream::fill_chunk(writable_memory_span chunk)
{
    ssize_t num_bytes_read = ::read(fd_.get(), chunk.data(), chunk.size());
    if (num_bytes_read == -1)
        throw_system_error(last_error(),
            "'{}' cannot be read", path_.string());

    return static_cast<std::size_t>(num_bytes_read);
}

std::size_t
file_stream::seek(std::size_t offset, bool from_current) const
{
    ::off_t off = ::lseek(
        fd_.get(), static_cast<::off_t>(offset), from_current ? SEEK_CUR : SEEK_SET);

    if (off != -1)
        return static_cast<std::size_t>(off);

    std::error_code err = last_error();

    if (err == std::errc::invalid_seek)
        throw_<byte_stream_error>("'{}' is not seekable.", path_.string());

    throw_system_error(err,
        "'{}' cannot be read", path_.string());
}

}  // namespace fairseq2n::detail
