// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <random>

namespace fairseq2n::detail {

inline std::uint64_t
pseudo_random()
{
    std::random_device rd{};

    return ((static_cast<std::uint64_t>(rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
}

}  // namespace fairseq2n::detail
