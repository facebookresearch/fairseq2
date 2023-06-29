// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "fairseq2/native/data/data_pipeline.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class zipped_data_source final : public data_source {
public:
    explicit
    zipped_data_source(
        std::vector<data_pipeline> &&pipelines,
        std::optional<std::vector<std::string>> &&names,
        bool warn_only,
        bool disable_parallelism) noexcept;

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    std::vector<data_pipeline> pipelines_;
    std::vector<std::string> names_;
    bool warn_only_;
    bool disable_parallelism_;
};

}  // namespace fairseq2::detail
