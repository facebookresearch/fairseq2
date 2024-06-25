// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class map_data_source final : public data_source {
public:
    explicit
    map_data_source(
        std::unique_ptr<data_source> &&inner,
        std::vector<map_fn> &&fns,
        std::size_t num_parallel_calls);

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
    bool
    fill_buffer();

    std::optional<data>
    invoke_function(data &&example, std::size_t fn_idx);

private:
    std::unique_ptr<data_source> inner_;
    std::vector<map_fn> map_fns_;
    std::size_t num_parallel_calls_;
    std::vector<std::optional<data>> buffer_{};
    std::vector<std::optional<data>>::iterator buffer_pos_{};
};

}  // namespace fairseq2n::detail
