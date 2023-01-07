// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/record_reader.h"

#include <algorithm>

#include "fairseq2/native/error.h"

using namespace fairseq2::detail;

namespace fairseq2 {

record_reader::~record_reader() = default;

memory_block
record_reader::next()
{
    if (!load_record())
        return {};

    memory_block record = extract_record();

    move_to_next_record();

    return record;
}

void
record_reader::reset()
{
    last_chunk_ = {};

    prev_chunks_.clear();

    stream_->reset();
}

bool
record_reader::load_record()
{
    record_size_ = 0;

    std::optional<std::size_t> record_offset = std::nullopt;

    while (!(record_offset = find_record_end(last_chunk_, prev_chunks_.empty()))) {
        memory_block next_chunk = stream_->read_chunk();
        if (next_chunk.empty()) {
            if (last_chunk_.empty())
                return false;

            throw record_error{"The stream ends with a partial record."};
        }

        record_size_ += last_chunk_.size();

        prev_chunks_.push_back(std::move(last_chunk_));

        last_chunk_ = std::move(next_chunk);
    }

    record_size_ += *record_offset;

    next_record_offset_ = *record_offset;

    return true;
}

memory_block
record_reader::extract_record()
{
    if (prev_chunks_.empty())
        return last_chunk_.share_first(record_size_);

    return copy_split_record();
}

memory_block
record_reader::copy_split_record()
{
    writable_memory_block record = allocate_memory(record_size_);

    auto iter = record.begin();

    std::for_each(prev_chunks_.begin(), prev_chunks_.end(), [&iter](const memory_block &i) {
        iter = std::copy(i.begin(), i.end(), iter);
    });

    std::copy(last_chunk_.begin(), last_chunk_.begin() + next_record_offset_, iter);

    return record;
}

void
record_reader::move_to_next_record()
{
    last_chunk_ = last_chunk_.share_slice(next_record_offset_);

    prev_chunks_.clear();
}

record_error::~record_error() = default;

}  // namespace fairseq2
