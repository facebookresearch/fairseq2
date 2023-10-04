// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class cat_data_source final : public data_source {
public:
    explicit
    cat_data_source(
        std::vector<data_pipeline> &&pipeline1,
        std::vector<data_pipeline> &&pipeline2);

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

    std::vector<data_pipeline>
    concatenate(
        std::vector<data_pipeline> &&pipeline1,
        std::vector<data_pipeline> &&pipeline2);

private:
    std::vector<data_pipeline> pipeline1_;
    std::vector<data_pipeline> pipeline2_;
};

}  // namespace fairseq2n::detail