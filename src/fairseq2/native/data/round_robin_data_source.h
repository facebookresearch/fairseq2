// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include "fairseq2/native/data/data_pipeline.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

/// @brief round robin on a list of datasources - the round always ends with the first element of the biggest datasource.
class round_robin_data_source final : public data_source {
public:
    explicit
    round_robin_data_source(std::vector<data_pipeline> &&pipelines, std::vector<float> &&probs = {}) noexcept
        : data_pipelines_(std::move(pipelines))
    {
        probs_ = std::move(probs);
        index_ = 0;
        epoch_done_ = std::vector<bool>(data_pipelines_.size(), false);
        pipelines_count_ = data_pipelines_.size();
    }

    std::optional<data>
    next() override;

    std::size_t
    skip(std::size_t num_examples) override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    std::vector<data_pipeline> data_pipelines_;
    std::vector<float> probs_;
    std::vector<bool> epoch_done_;
    std::size_t index_;
    std::size_t pipelines_count_;

    std::vector<std::size_t> get_split_by_pipeline(std::size_t num_examples) const;
    void skip_elements_for_pipeline(std::size_t pipeline_index, std::size_t num_examples);
    void reset_pipeline(std::size_t pipeline_index);
    bool all_datasources_done();
};

}  // namespace fairseq2::detail
