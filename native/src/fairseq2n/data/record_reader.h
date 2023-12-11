// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fairseq2n/api.h"
#include "fairseq2n/memory.h"
#include "fairseq2n/data/byte_stream.h"

namespace fairseq2n {

class FAIRSEQ2_API record_reader {
protected:
    explicit
    record_reader(std::unique_ptr<byte_stream> &&stream) noexcept
      : stream_{std::move(stream)}
    {}

public:
    record_reader(const record_reader &) = delete;
    record_reader &operator=(const record_reader &) = delete;

    record_reader(record_reader &&) = default;
    record_reader &operator=(record_reader &&) = default;

    virtual
   ~record_reader();

    memory_block
    next();

    void
    reset();

private:
    bool
    load_next_record();

    virtual std::optional<std::size_t>
    maybe_find_record_end(memory_span chunk, bool first_chunk) = 0;

    memory_block
    extract_record();

    memory_block
    copy_split_record();

    void
    move_to_next_record();

private:
    std::unique_ptr<byte_stream> stream_;
    memory_block current_chunk_{};
    std::vector<memory_block> previous_chunks_{};
    std::size_t record_len_ = 0;
    std::size_t record_end_offset_ = 0;
};

class FAIRSEQ2_API record_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

public:
    record_error(const record_error &) = default;
    record_error &operator=(const record_error &) = default;

   ~record_error() override;
};

}  // namespace fairseq2n
