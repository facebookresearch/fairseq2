// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"
#include "fairseq2n/data/composite_data_source.h"

namespace fairseq2n::detail {

class round_robin_data_source final : public data_source {
public:
    explicit
    round_robin_data_source(std::vector<data_pipeline> &&pipelines, bool stop_at_shortest);

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    std::optional<data>
    next_in_pipeline(std::size_t pipeline_idx);

private:
    std::unique_ptr<composite_data_source> inner_;
    std::size_t pipeline_idx_;
    std::size_t pipelines_count_;
};

}  // namespace fairseq2n::detail
