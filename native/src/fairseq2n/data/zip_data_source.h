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

class zip_data_source final : public data_source {
public:
    explicit
    zip_data_source(
        std::vector<data_pipeline> &&pipelines,
        std::vector<std::string> &&names,
        bool zip_to_shortest,
        bool flatten,
        bool disable_parallelism);

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
    static std::optional<data>
    flatten_to_dict(data_list &zip);

    static std::optional<data>
    flatten_to_list(data_list &zip);

private:
    std::vector<data_pipeline> pipelines_;
    std::vector<std::string> names_;
    bool zip_to_shortest_;
    bool flatten_;
    bool disable_parallelism_;
    data_source_finitude_type finitude_type_;
};

}  // namespace fairseq2n::detail
