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
    mapped_data_source(std::unique_ptr<data_source> &&inner, map_fn &&fn, std::size_t chunk_size) noexcept
        : inner_{std::move(inner)}, fn_{std::move(fn)}, chunk_size_(chunk_size)
    {
        buffer_.reserve(chunk_size);
        buffer_iter_ = buffer_.begin();
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
    void map_at(std::size_t i);
    std::unique_ptr<data_source> inner_;
    map_fn fn_;
    std::size_t chunk_size_;
    std::vector<data> buffer_{};
    std::vector<fairseq2::data>::iterator buffer_iter_;
};

}  // namespace fairseq2::detail
