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

    bool
    is_infinite() const noexcept override;

private:
    std::optional<data>
    next_in_pipeline(std::size_t pipeline_idx);

    bool
    are_all_done() noexcept;

private:
    std::vector<data_pipeline> pipelines_;
    std::vector<std::optional<data>> buffer_{};
    std::size_t buffer_idx_ = 0;
    std::vector<bool> is_epoch_done_;
    bool is_eod_ = false;
    bool stop_at_shortest_;
    bool is_infinite_;
};

}  // namespace fairseq2n::detail
