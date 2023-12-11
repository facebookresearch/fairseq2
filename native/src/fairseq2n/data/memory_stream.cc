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
memory_stream::reset()
{
    block_ = original_block_;
}

}  // namespace fairseq2n::detail
