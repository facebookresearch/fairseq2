// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <string>

#include "fairseq2/native/memory.h"
#include "fairseq2/native/data/stream.h"
#include "fairseq2/native/data/detail/file.h"

namespace fairseq2::detail {

class file_stream final : public stream {
public:
    explicit
    file_stream(file_desc &&fd, std::string pathname, std::size_t chunk_size) noexcept;

    memory_block
    read_chunk() override;

    void
    reset() override;

private:
    void
    hint_sequential_file() noexcept;

    std::size_t
    fill_chunk(writable_memory_span chunk);

private:
    file_desc fd_;
    std::string pathname_;
    std::size_t chunk_size_;
    bool eod_ = false;
};

}  // namespace fairseq2::detail
