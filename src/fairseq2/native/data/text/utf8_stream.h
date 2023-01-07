// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <iconv.h>

#include "fairseq2/native/memory.h"
#include "fairseq2/native/data/stream.h"

namespace fairseq2::detail {

class utf8_stream final : public stream {
    enum class iconv_status {
        ok,
        input_too_big,
        incomplete_sequence
    };

public:
    explicit
    utf8_stream(std::unique_ptr<stream> &&inner, std::string encoding, std::size_t chunk_size) noexcept;

    utf8_stream(const utf8_stream &) = delete;
    utf8_stream &operator=(const utf8_stream &) = delete;

    utf8_stream(utf8_stream &&) = delete;
    utf8_stream &operator=(utf8_stream &&) = delete;

   ~utf8_stream() override;

    memory_block
    read_chunk() override;

    void
    reset() override;

private:
    bool
    is_utf8_encoding() const noexcept
    {
        return encoding_ == "UTF-8" || encoding_ == "utf-8";
    }

    memory_block
    read_chunk_core();

    bool
    load_next_inner_chunk();

    void
    move_leftover_bits();

    iconv_status
    decode_inner_chunk(writable_memory_span &out);

    void
    ensure_iconv_initialized();

    void
    reset_iconv() noexcept;

private:
    static const ::iconv_t invalid_iconv_;

    std::unique_ptr<stream> inner_;
    std::string encoding_;
    bool is_utf8_;
    std::size_t chunk_size_;
    ::iconv_t iconv_{invalid_iconv_};
    memory_block inner_chunk_{};
    memory_block leftover_bits_{};
    bool eod_ = false;
};

}  // namespace fairseq2::detail

