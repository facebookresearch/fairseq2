// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class FAIRSEQ2_API string_to_integer_converter final : public data_processor {
public:
    explicit
    string_to_integer_converter(std::int16_t base = 10) noexcept
      : base_{base}
    {}

    data
    process(data &&d) const override;

private:
    std::int16_t base_;
};

}  // namespace fairseq2
