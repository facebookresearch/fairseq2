// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#pragma once

#include "fairseq2/native/data/data_pipeline.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class filtered_data_source final : public data_source {
public :
    explicit
    filtered_data_source(std::unique_ptr<data_source> &&inner, predicate_fn &&predicate) noexcept
        : inner_{std::move(inner)}, predicate_{std::move(predicate)}
    {
    }

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

private:
    std::unique_ptr<data_source> inner_;
    predicate_fn predicate_;

    bool
    try_predicate(data &value);
};

} // namespace fairseq2::detail
