// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include "fairseq2n/data/data_pipeline.h"


using index_generator_fn = std::function<std::size_t()>;

namespace fairseq2n::detail {

class composite_data_source final : public data_source {
public:
    explicit
    composite_data_source(std::vector<data_pipeline> &&pipelines, index_generator_fn &&index_gen_fn, bool stop_at_shortest);

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

    bool
    eod();

private:
    std::vector<data_pipeline> pipelines_;
    index_generator_fn next_index_gen_;
    std::vector<std::optional<data>> buffer_{};
    std::vector<bool> is_epoch_done_;
    bool is_eod_ = false;
    bool stop_at_shortest_;
};

}  // namespace fairseq2n::detail
