// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class dynamic_bucket_data_source final : public data_source {
public:
    explicit
    dynamic_bucket_data_source(
        std::unique_ptr<data_source> &&inner,
        float64 threshold,
        cost_fn &&fn,
        std::optional<std::size_t> maybe_nb_min,
        std::optional<std::size_t> maybe_nb_max,
        bool drop_remainder) noexcept;

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
    float64
    invoke_function(data &example);

private:
    std::unique_ptr<data_source> inner_;
    float64 threshold_;
    cost_fn cost_fn_;
    std::optional<std::size_t> maybe_nb_min_;
    std::optional<std::size_t> maybe_nb_max_;
    bool drop_remainder_;
};

}  // namespace fairseq2n::detail
