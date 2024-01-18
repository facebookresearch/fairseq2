// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <utility>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class yield_from_data_source final : public data_source {
public:
    explicit
    yield_from_data_source(std::unique_ptr<data_source> &&inner, yield_fn &&fn) noexcept
      : inner_{std::move(inner)}, yield_fn_{std::move(fn)}
    {}

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
    bool
    load_next_data_pipeline();

    data_pipeline
    invoke_function(data &example);

private:
    std::unique_ptr<data_source> inner_;
    yield_fn yield_fn_;
    std::optional<data> maybe_current_example_{};
    data_pipeline data_pipeline_{};
};

}  // namespace fairseq2n::detail
