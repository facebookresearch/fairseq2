// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/dynamic_bucket_data_source.h"

#include <vector>

namespace fairseq2n::detail {

dynamic_bucket_data_source::dynamic_bucket_data_source(
    std::unique_ptr<data_source> &&inner, float64 threshold, cost_fn &&fn, std::optional<std::size_t> maybe_nb_min, std::optional<std::size_t> maybe_nb_max, bool drop_remainder) noexcept
  : inner_{std::move(inner)}, threshold_{threshold}, cost_fn_{std::move(fn)}, maybe_nb_min_{maybe_nb_min}, maybe_nb_max_{maybe_nb_max}, drop_remainder_{drop_remainder}
{}

std::optional<data>
dynamic_bucket_data_source::next()
{
    data_list output{};

    if (maybe_nb_min_)
        output.reserve(*maybe_nb_min_);

    float64 cost = 0;

    auto bucket_ready = [&]() {
        bool cost_threshold_met = (cost >= threshold_);
        bool minimum_size_met = (maybe_nb_min_ ? (output.size() >= *maybe_nb_min_) : true);
        bool maximum_size_met = (maybe_nb_max_ ? (output.size() >= *maybe_nb_max_) : false);
        return maximum_size_met || (cost_threshold_met && minimum_size_met);
    };

    while (!bucket_ready()) {
        std::optional<data> maybe_example = inner_->next();
        if (!maybe_example)
            break;
        cost += invoke_function(*maybe_example);
        output.push_back(*std::move(maybe_example));
    }

    if (output.empty())
        return std::nullopt;

    if (drop_remainder_ && !bucket_ready())
        return std::nullopt;

    return output;
}

void
dynamic_bucket_data_source::reset(bool reset_rng)
{
    inner_->reset(reset_rng);
}

void
dynamic_bucket_data_source::record_position(tape &t, bool strict) const
{
    inner_->record_position(t, strict);
}

void
dynamic_bucket_data_source::reload_position(tape &t, bool strict)
{
    inner_->reload_position(t, strict);
}

data_source_finitude_type
dynamic_bucket_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

float64
dynamic_bucket_data_source::invoke_function(data &example)
{
    return cost_fn_(example);
}

}  // namespace fairseq2n::detail
