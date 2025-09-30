// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <utility>

#include "fairseq2n/memory.h"
#include "fairseq2n/data/byte_stream.h"

namespace fairseq2n::detail {

class memory_stream final : public byte_stream {
public:
    explicit
    memory_stream(memory_block block) noexcept
      : block_{std::move(block)}
    {
        original_block_ = block_;
    }

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
    memory_block block_;
    memory_block original_block_;
};

}  // namespace fairseq2n::detail
