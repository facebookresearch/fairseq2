// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "fairseq2n/memory.h"
#include "fairseq2n/data/byte_stream.h"
#include "fairseq2n/data/detail/file.h"

namespace fairseq2n::detail {

class file_stream final : public byte_stream {
public:
    explicit
    file_stream(file_desc &&fd, std::filesystem::path path, std::size_t chunk_size) noexcept;

private:
    void
    hint_sequential_file() noexcept;

public:
    memory_block
    read_chunk() override;

    void
    seek(std::size_t offset) override;

    std::size_t
    position() const override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

    bool
    supports_seek() const noexcept override;

private:
    std::size_t
    fill_chunk(writable_memory_span chunk);

    std::size_t
    seek(std::size_t offset, bool from_current) const;

private:
    file_desc fd_;
    std::filesystem::path path_;
    std::size_t chunk_size_;
    bool is_eod_ = false;
};

}  // namespace fairseq2n::detail
