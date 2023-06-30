// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "fairseq2/native/data/data_processor.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2::detail {

class mapped_data_source final : public data_source {
public:
    explicit
    mapped_data_source(
        std::unique_ptr<data_source> &&inner,
        std::shared_ptr<const data_processor> &&p,
        std::size_t num_parallel_calls,
        bool warn_only) noexcept
      : inner_{std::move(inner)},
        processor_{std::move(p)},
        num_parallel_calls_{num_parallel_calls},
        warn_only_{warn_only}
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

    std::optional<data>
    invoke_processor(data &&d);

private:
    std::unique_ptr<data_source> inner_;
    std::shared_ptr<const data_processor> processor_;
    std::size_t num_parallel_calls_;
    bool warn_only_;
    std::vector<std::optional<data>> buffer_{};
    std::vector<std::optional<data>>::iterator buffer_iter_{};
};

}  // namespace fairseq2::detail
