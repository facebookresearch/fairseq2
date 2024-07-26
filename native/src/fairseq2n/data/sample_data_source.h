// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <ATen/Generator.h>

#include "fairseq2n/float.h"
#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class sample_data_source final : public data_source {
public:
    explicit
    sample_data_source(
        std::vector<data_pipeline> &&pipelines,
        std::vector<float32> &&weights,
        std::optional<std::uint64_t> maybe_seed,
        bool allow_repeats);

    std::optional<data>
    next() override;

    void
    reset(bool reset_rng) override;

    void
    record_position(tape &t, bool strict) const override;

    void
    reload_position(tape &t, bool strict) override;

    data_source_finitude_type
    finitude_type() const noexcept override;

private:
    std::size_t
    random_pipeline_index();

    std::optional<data>
    next_in_pipeline(std::size_t pipeline_idx);

    bool
    are_all_done() noexcept;

    void block(std::size_t idx);

    void sum_weights();

private:
    std::vector<data_pipeline> pipelines_;
    std::vector<float32> original_weight_cumsums_;
    std::vector<float32> weight_cumsums_;
    std::vector<std::optional<data>> buffer_{};
    std::vector<bool> is_epoch_done_;
    bool is_eod_ = false;
    data_source_finitude_type finitude_type_;
    std::uint64_t seed_;
    bool allow_repeats_;
    at::Generator generator_;
};

}  // namespace fairseq2n::detail
