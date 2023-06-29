// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2 {

class FAIRSEQ2_API data_processor {
public:
    data_processor() noexcept = default;

    data_processor(const data_processor &) noexcept = default;
    data_processor &operator=(const data_processor &) noexcept = default;

    data_processor(data_processor &&) noexcept = default;
    data_processor &operator=(data_processor &&) noexcept = default;

    virtual
   ~data_processor();

    virtual data
    process(data &&) const = 0;
};

}  // namespace fairseq2
