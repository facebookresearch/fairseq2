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

#include "fairseq2/native/api.h"
#include "fairseq2/native/memory.h"
#include "fairseq2/native/data/stream.h"

namespace fairseq2 {

class FAIRSEQ2_API record_reader {
protected:
    explicit
    record_reader(std::unique_ptr<stream> &&s) noexcept
      : stream_{std::move(s)}
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
    load_record();

    virtual std::optional<std::size_t>
    find_record_end(memory_span chunk, bool first_chunk) = 0;

    memory_block
    extract_record();

    memory_block
    copy_split_record();

    void
    move_to_next_record();

private:
    std::unique_ptr<stream> stream_;
    memory_block last_chunk_{};
    std::vector<memory_block> prev_chunks_{};
    std::size_t record_size_ = 0;
    std::size_t next_record_offset_ = 0;
};

class FAIRSEQ2_API record_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

public:
    record_error(const record_error &) = default;
    record_error &operator=(const record_error &) = default;

   ~record_error() override;
};

}  // namespace fairseq2
