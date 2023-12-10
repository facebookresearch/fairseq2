// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include <sndfile.h>

#include "fairseq2n/float.h"
#include "fairseq2n/memory.h"
#include "fairseq2n/span.h"
#include "fairseq2n/utils/cast.h"

namespace fairseq2n::detail {

// Wraps `memory_block` as a "virtual" file to use with libsndfile.
class vio_file {
public:
    explicit
    vio_file(memory_block &&block) noexcept
      : block_{std::move(block)}
    {}

    ::sf_count_t
    seek(::sf_count_t offset, int whence) noexcept;

    ::sf_count_t
    read(void *ptr, ::sf_count_t count) noexcept;

    ::sf_count_t
    write(const void *ptr, ::sf_count_t count);

    ::sf_count_t
    tell() const noexcept
    {
        return current_pos_;
    }

    ::sf_count_t
    size() const noexcept
    {
        return conditional_cast<::sf_count_t>(block_.size());
    }

private:
    static std::size_t
    as_size(::sf_count_t value) noexcept
    {
        return conditional_cast<std::size_t>(value);
    }

private:
    memory_block block_;
    ::sf_count_t current_pos_{};
};

class sndfile {
public:
    static sndfile
    from_memory(memory_block block);

private:
    explicit
    sndfile(::SNDFILE *handle, ::SF_INFO audio_info, std::unique_ptr<vio_file> &&file) noexcept
      : handle_{handle}, audio_info_{audio_info}, file_{std::move(file)}
    {}

public:
    sndfile() noexcept = default;

    sndfile(const sndfile &) = delete;
    sndfile &operator=(const sndfile &) = delete;

    sndfile(sndfile &&other) noexcept
      : handle_{other.handle_}, audio_info_{other.audio_info_}, file_{std::move(other.file_)}
    {
        other.handle_ = nullptr;

        other.audio_info_ = {};
    }

    sndfile &
    operator=(sndfile &&other) noexcept
    {
        handle_ = std::exchange(other.handle_, nullptr);

        audio_info_ = std::exchange(other.audio_info_, {});

        file_ = std::move(other.file_);

        return *this;
    }

   ~sndfile();

    void
    decode_into(span<float32> target);

    void
    decode_into(span<std::int16_t> target);

    void
    decode_into(span<std::int32_t> target);

    std::int64_t
    num_frames() const noexcept
    {
        return conditional_cast<std::int64_t>(audio_info_.frames);
    }

    std::int64_t
    num_channels() const noexcept
    {
        return conditional_cast<std::int64_t>(audio_info_.channels);
    }

    std::int64_t
    sample_rate() const noexcept
    {
        return conditional_cast<std::int64_t>(audio_info_.samplerate);
    }

    std::int64_t
    format() const noexcept
    {
        return conditional_cast<std::int64_t>(audio_info_.format);
    }

private:
    void
    check_handle()
    {
        check_handle(handle_);
    }

    static void
    check_handle(::SNDFILE *handle);

private:
    ::SNDFILE *handle_{};
    ::SF_INFO audio_info_{};
    std::unique_ptr<vio_file> file_{};
};

}  // namespace fairseq2n::detail
