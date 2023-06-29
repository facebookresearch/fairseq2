// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class FAIRSEQ2_API composite_data_processor final : public data_processor {
public:
    explicit
    composite_data_processor(std::vector<std::shared_ptr<const data_processor>> procs) noexcept
      : processors_{std::move(procs)}
    {}

    data
    process(data &&d) const override;

private:
    std::vector<std::shared_ptr<const data_processor>> processors_;
};

}  // namespace fairseq2

