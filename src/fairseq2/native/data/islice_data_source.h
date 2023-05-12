// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2/native/data/data_source.h"
#include <stdexcept>

namespace fairseq2::detail {

class islice_data_source final : public data_source {
public :
    explicit
    islice_data_source(std::unique_ptr<data_source> &&inner, std::optional<std::size_t> start, std::optional<std::size_t> stop, std::optional<std::size_t> step)
        : inner_{std::move(inner)}, stop_{stop}
    {
        start_ = start.value_or(0);
        step_ = step.value_or(1);
        next_index_ = 0;

        if (stop_ && stop_.value() < start)
            throw std::invalid_argument("stop value should always be greater than start value.");
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
    std::unique_ptr<data_source> inner_;
    std::size_t start_;
    std::optional<std::size_t> stop_;
    std::size_t step_;
    std::size_t next_index_;
};

}  // namespace fairseq2::detail
