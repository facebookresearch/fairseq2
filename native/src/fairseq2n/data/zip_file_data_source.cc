// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/zip_file_data_source.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <system_error>
#include <utility>

#include <zip/src/zip.h>

#include "fairseq2n/memory.h"
#include "fairseq2n/data/byte_stream.h"
#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/file.h"
#include "fairseq2n/data/immutable_string.h"
#include "fairseq2n/detail/exception.h"
#include "fairseq2n/utils/string.h"

namespace fairseq2n::detail {

zip_file_data_source::zip_file_data_source(std::string &&pathname)
    : pathname_{std::move(pathname)}
{
    try {
        zip_reader_ = zip_open(pathname_.c_str(), ZIP_DEFAULT_COMPRESSION_LEVEL, 'r');
        num_entries_ = (std::size_t)zip_entries_total(zip_reader_);
    } catch (const std::exception &) {
        handle_error();
    }
}

std::optional<data>
zip_file_data_source::next()
{
    if (num_files_read_ >= num_entries_) return std::nullopt;

    fairseq2n::writable_memory_block zip_entry;
    zip_entry_openbyindex(zip_reader_, num_files_read_);
    {
        auto size = zip_entry_size(zip_reader_);
        zip_entry = fairseq2n::allocate_memory(size);
        zip_entry_noallocread(zip_reader_, (void *)zip_entry.data(), size);
    }
    zip_entry_close(zip_reader_);

    num_files_read_ += 1;
    return immutable_string{zip_entry};
}

void
zip_file_data_source::reset()
{
    num_files_read_ = 0;
}

void
zip_file_data_source::record_position(tape &t) const
{
    t.record(num_files_read_);
}

void
zip_file_data_source::reload_position(tape &t)
{
    auto num_files_read = t.read<std::size_t>();

    reset();

    // TODO: use random access
    for (std::size_t i = 0; i < num_files_read; i++)
        next();
}

bool
zip_file_data_source::is_infinite() const noexcept
{
    return false;
}

void
zip_file_data_source::handle_error()
{
    try {
        throw;
    } catch (const byte_stream_error &) {
        throw_read_failure();
    } catch (const std::system_error &) {
        throw_read_failure();
    }
}

inline void
zip_file_data_source::throw_read_failure()
{
    throw_with_nested<data_pipeline_error>(
        "The data pipeline cannot read from '{}'. See nested exception for details.", pathname_);
}

}  // namespace fairseq2n::detail
