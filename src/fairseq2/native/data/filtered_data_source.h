// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <utility>

#include "fairseq2/native/data/data_pipeline.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class filtered_data_source final : public data_source {
public :
    explicit
    filtered_data_source(std::unique_ptr<data_source> &&inner, predicate_fn &&f) noexcept
      : inner_{std::move(inner)}, fn_{std::move(f)}
    {}

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    bool
    invoke_fn(data &d);

private:
    std::unique_ptr<data_source> inner_;
    predicate_fn fn_;
};

} // namespace fairseq2::detail
