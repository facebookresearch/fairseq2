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
        bool disable_parallelism) noexcept;

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
};

}  // namespace fairseq2n::detail
