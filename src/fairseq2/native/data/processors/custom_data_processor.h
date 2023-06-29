// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>
#include <utility>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class FAIRSEQ2_API custom_data_processor final : public data_processor {
public:
    explicit
    custom_data_processor(std::function<data(data &&)> f) noexcept
      : fn_{std::move(f)}
    {}

    data
    process(data &&d) const override;

private:
    std::function<data(data &&)> fn_;
};

}  // namespace fairseq2
