// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <utility>

#include "fairseq2/native/memory.h"
#include "fairseq2/native/data/stream.h"

namespace fairseq2::detail {

class memory_stream final : public stream {
public:
    explicit
    memory_stream(memory_block b) noexcept
      : block_{std::move(b)}
    {
        original_block_ = block_;
    }

    memory_block
    read_chunk() override;

    void
    reset() override;

private:
    memory_block block_;
    memory_block original_block_;
};

}  // namespace fairseq2::detail
