// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

namespace fairseq2n {

class FAIRSEQ2_API string_to_int_converter final {
public:
    explicit
    string_to_int_converter(std::int16_t base = 10) noexcept
      : base_{base}
    {}

    data
    operator()(data &&d) const;

private:
    std::int16_t base_;
};

}  // namespace fairseq2n
