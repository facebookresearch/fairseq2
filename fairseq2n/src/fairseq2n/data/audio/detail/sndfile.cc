// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/audio/detail/sndfile.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

::sf_count_t
vio_file::seek(::sf_count_t offset, int whence) noexcept
{
    switch (whence) {
    case SEEK_SET:
        current_pos_ = offset;
        break;
    case SEEK_CUR:
    case SEEK_END:
        current_pos_ += offset;
        break;
    }

    current_pos_ = std::max(std::min(size(), current_pos_), {});

    return current_pos_;
}

::sf_count_t
vio_file::read(void *ptr, ::sf_count_t count) noexcept
{
    memory_span source = block_;

    count = std::min(size() - current_pos_, count);

    source = source.subspan(as_size(current_pos_), as_size(count));

    std::copy(source.begin(), source.end(), static_cast<std::byte *>(ptr));

    current_pos_ += count;

    return count;
}

::sf_count_t
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
vio_file::write(const void *, ::sf_count_t)
{
    // We only support decoding audio files.
    return -1;
}

namespace {

::sf_count_t
vio_get_filelen(void *user_data)
{
    return static_cast<vio_file *>(user_data)->size();
}

::sf_count_t
vio_seek(::sf_count_t offset, int whence, void *user_data)
{
    return static_cast<vio_file *>(user_data)->seek(offset, whence);
}

::sf_count_t
vio_read(void *ptr, ::sf_count_t count, void *user_data)
{
    return static_cast<vio_file *>(user_data)->read(ptr, count);
}

::sf_count_t
vio_write(const void *ptr, ::sf_count_t count, void *user_data)
{
    return static_cast<vio_file *>(user_data)->write(ptr, count);
}

::sf_count_t
vio_tell(void *user_data)
{
    return static_cast<vio_file *>(user_data)->tell();
}

// Virtual I/O dispatch table for libsndfile.
::SF_VIRTUAL_IO vio_dispatch_table{vio_get_filelen, vio_seek, vio_read, vio_write, vio_tell};

}  // namespace

sndfile
sndfile::from_memory(memory_block block)
{
    ::SF_INFO info{};

    // Wrap `block` as a "virtual" file.
    auto file = std::make_unique<vio_file>(std::move(block));

    ::SNDFILE *handle = sf_open_virtual(&vio_dispatch_table, ::SFM_READ, &info, file.get());

    check_handle(handle);

    return sndfile{handle, info, std::move(file)};
}

sndfile::~sndfile()
{
    if (handle_ != nullptr)
        ::sf_close(handle_);
}

void
sndfile::decode_into(span<float32> target)
{
    ::sf_count_t num_frames_decoded = ::sf_readf_float(handle_, target.data(), audio_info_.frames);

    if (num_frames_decoded != audio_info_.frames)
        throw_<internal_error>(
            "`sndfile` has failed to decode the input audio. Please file a bug report.");
}

void
sndfile::decode_into(span<std::int32_t> target)
{
    static_assert(sizeof(int) == sizeof(std::int32_t),
        "The host platform's `int` data type must be 32-bit.");

    ::sf_count_t num_frames_decoded = ::sf_readf_int(handle_, target.data(), audio_info_.frames);

    if (num_frames_decoded != audio_info_.frames)
        throw_<internal_error>(
            "`sndfile` has failed to decode the input audio. Please file a bug report.");
}

void
sndfile::decode_into(span<std::int16_t> target)
{
    static_assert(sizeof(short int) == sizeof(std::int16_t),
        "The host platform's `short int` data type must be 16-bit.");

    ::sf_count_t num_frames_decoded = ::sf_readf_short(handle_, target.data(), audio_info_.frames);

    if (num_frames_decoded != audio_info_.frames)
        throw_<internal_error>(
            "`sndfile` has failed to decode the input audio. Please file a bug report.");
}

void
sndfile::check_handle(::SNDFILE *handle)
{
    int err_num = ::sf_error(handle);
    if (err_num == ::SF_ERR_NO_ERROR)
        return;

    const char *err_msg = ::sf_error_number(err_num);
    if (err_msg == nullptr)
        err_msg = "An unknown libsndfile error has occurred.";

    if (err_num == ::SF_ERR_SYSTEM || err_num == ::SF_ERR_UNSUPPORTED_ENCODING)
        throw_<std::runtime_error>(err_msg);
    else
        throw_<std::invalid_argument>(err_msg);
}

}  // namespace fairseq2n::detail
