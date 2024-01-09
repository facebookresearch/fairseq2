// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/bucket_data_source.h"

#include <vector>

namespace fairseq2n::detail {

bucket_data_source::bucket_data_source(
    std::unique_ptr<data_source> &&inner, std::size_t bucket_size, bool drop_remainder) noexcept
  : inner_{std::move(inner)}, bucket_size_{bucket_size}, drop_remainder_{drop_remainder}
{}

std::optional<data>
bucket_data_source::next()
{
    data_list output{};

    output.reserve(bucket_size_);

    for (std::size_t i = 0; i < bucket_size_; ++i) {
        std::optional<data> maybe_example = inner_->next();
        if (!maybe_example)
            break;

        output.push_back(*std::move(maybe_example));
    }

    if (output.empty())
        return std::nullopt;

    if (drop_remainder_ && output.size() < bucket_size_)
        return std::nullopt;

    return output;
}

void
bucket_data_source::reset()
{
    inner_->reset();
}

void
bucket_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
bucket_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

bool
bucket_data_source::is_infinite() const noexcept
{
    return inner_->is_infinite();
}

}  // namespace fairseq2n::detail
