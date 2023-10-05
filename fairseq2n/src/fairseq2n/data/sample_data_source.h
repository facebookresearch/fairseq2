// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <vector>

#include <ATen/Generator.h>
#include <ATen/Tensor.h>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"
#include "fairseq2n/data/composite_data_source.h"

namespace fairseq2n::detail {

/// @brief sample from a list of datasources
class sample_data_source final : public data_source {
public:
    explicit
    sample_data_source(std::vector<data_pipeline> &&pipelines, std::vector<float32> &&weights, bool stop_at_shortest);

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    std::size_t
    next_index();

private:
    std::unique_ptr<composite_data_source> inner_;

    at::Generator generator_;
    at::Tensor weights_;
};

}  // namespace fairseq2::detail
