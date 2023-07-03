// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <optional>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class FAIRSEQ2_API collater : public data_processor {
public:
    explicit
    collater(std::optional<std::int64_t> pad_idx) noexcept
      : pad_idx_{pad_idx}
    {}

    data
    process(data &&d) const override;

private:
    std::optional<std::int64_t> pad_idx_;
};

}  // namespace fairseq2
