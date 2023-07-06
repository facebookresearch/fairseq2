// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <optional>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2 {

class FAIRSEQ2_API collater {
public:
    explicit
    collater(std::optional<std::int64_t> pad_idx) noexcept
      : pad_idx_{pad_idx}
    {}

    data
    operator()(data &&d) const;

private:
    std::optional<std::int64_t> pad_idx_;
};

}  // namespace fairseq2
