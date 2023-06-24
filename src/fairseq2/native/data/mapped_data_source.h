// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "fairseq2/native/data/data_pipeline.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class mapped_data_source final : public data_source {
public:
    explicit
    mapped_data_source(
        std::unique_ptr<data_source> &&inner,
        map_fn &&fn,
        std::size_t num_parallel_calls) noexcept
      : inner_{std::move(inner)}, fn_{std::move(fn)}, num_parallel_calls_{num_parallel_calls}
    {
        buffer_.reserve(num_parallel_calls);

        buffer_iter_ = buffer_.begin();
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
    bool
    fill_buffer();

    data
    invoke_fn(data &&example);

private:
    std::unique_ptr<data_source> inner_;
    map_fn fn_;
    std::size_t num_parallel_calls_;
    std::vector<data> buffer_{};
    std::vector<data>::iterator buffer_iter_{};
};

}  // namespace fairseq2::detail
