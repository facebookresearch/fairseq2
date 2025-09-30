// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/memory_stream.h"

namespace fairseq2n::detail {

memory_block
memory_stream::read_chunk()
{
    return std::exchange(block_, {});
}

void
memory_stream::seek(std::size_t offset)
{
    if (offset >= block_.size())
        block_ = {};
    else
        block_ = block_.share_slice(offset);
}

std::size_t
memory_stream::position() const
{
    return original_block_.size() - block_.size();
}

void
memory_stream::reset()
{
    block_ = original_block_;
}

void
memory_stream::record_position(tape &t) const
{
    std::size_t offset = position();

    t.record(offset);
}

void
memory_stream::reload_position(tape &t)
{
    block_ = original_block_;

    seek(t.read<std::size_t>());
}

bool
memory_stream::supports_seek() const noexcept
{
    return true;
}

}  // namespace fairseq2n::detail
