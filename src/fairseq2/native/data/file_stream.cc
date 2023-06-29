// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/file_stream.h"

#include <utility>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

#include <fmt/core.h>

#include "fairseq2/native/error.h"

namespace fairseq2::detail {

file_stream::file_stream(file_desc &&fd, std::string pathname, std::size_t chunk_size) noexcept
  : fd_{std::move(fd)}, pathname_{std::move(pathname)}, chunk_size_{chunk_size}
{
    hint_sequential_file();
}

memory_block
file_stream::read_chunk()
{
    if (eod_)
        return {};

    writable_memory_block chunk = allocate_memory(chunk_size_);

    writable_memory_span s = chunk;

    while (!s.empty()) {
        std::size_t num_bytes_read = fill_chunk(s);
        if (num_bytes_read == 0) {
            eod_ = true;

            break;
        }

        s = s.subspan(num_bytes_read);
    }

    if (s.empty())
        return chunk;

    return chunk.share_first(chunk.size() - s.size());
}

void
file_stream::reset()
{
    ::off_t o = ::lseek(fd_.get(), 0, SEEK_SET);
    if (o == -1) {
        std::error_code err = last_error();

        if (err == std::errc::invalid_seek)
            throw stream_error{
                fmt::format("'{}' cannot be reset since it is not seekable.", pathname_)};

        throw std::system_error{err,
            fmt::format("'{}' cannot be reset", pathname_)};
    }

    eod_ = false;
}

void
file_stream::hint_sequential_file() noexcept
{
#ifdef __linux__
    int r = ::posix_fadvise(fd_.get(), 0, 0, POSIX_FADV_SEQUENTIAL);
    if (r != 0) {
        // TODO: warn
    }
#endif
}

std::size_t
file_stream::fill_chunk(writable_memory_span chunk)
{
    ssize_t num_bytes_read = ::read(fd_.get(), chunk.data(), chunk.size());
    if (num_bytes_read == -1)
        throw std::system_error{last_error(),
            fmt::format("'{}' cannot be read", pathname_)};

    return static_cast<std::size_t>(num_bytes_read);
}

}  // namespace fairseq2::detail
