// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <utility>

#include "fairseq2/native/py.h"
#include "fairseq2/native/data/data_pipeline.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class mapped_data_source final : public data_source {
public:
    explicit
    mapped_data_source(std::unique_ptr<data_source> &&inner, map_fn &&fn) noexcept
        : inner_{std::move(inner)}, fn_{std::move(fn)}
    {}

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
    map_fn fn_;
};

}  // namespace fairseq2::detail
