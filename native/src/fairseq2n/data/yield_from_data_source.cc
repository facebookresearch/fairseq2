// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/yield_from_data_source.h"

#include <exception>

#include "fairseq2n/data/detail/exception.h"

namespace fairseq2n::detail {

std::optional<data>
yield_from_data_source::next()
{
    std::optional<data> maybe_example{};

    while (!(maybe_example = data_pipeline_.next()))
        if (!load_next_data_pipeline())
            break;

    return maybe_example;
}

void
yield_from_data_source::reset(bool reset_rng)
{
    maybe_current_example_ = {};

    data_pipeline_ = {};

    inner_->reset(reset_rng);
}

void
yield_from_data_source::record_position(tape &t, bool strict) const
{
    if (strict) {
        t.record(maybe_current_example_);

        if (maybe_current_example_)
            data_pipeline_.record_position(t, strict);
    }

    inner_->record_position(t, strict);
}

void
yield_from_data_source::reload_position(tape &t, bool strict)
{
    if (strict) {
        maybe_current_example_ = t.read<std::optional<data>>();

        if (maybe_current_example_) {
            // The assumption we make is that the recorded example will fully
            // reconstruct the original yielded-from data pipeline.
            data_pipeline_ = invoke_function(*maybe_current_example_);

            // With that assumption, we restore the recorded state.
            data_pipeline_.reload_position(t);
        }
        else
            data_pipeline_ = {};
    } else {
        data_pipeline_ = {};

        maybe_current_example_ = std::nullopt;
    }

    inner_->reload_position(t, strict);
}

data_source_finitude_type
yield_from_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

bool
yield_from_data_source::load_next_data_pipeline()
{
    maybe_current_example_ = inner_->next();

    if (maybe_current_example_)
        data_pipeline_ = invoke_function(*maybe_current_example_);
    else
        data_pipeline_ = {};

    if (data_pipeline_.finitude_type() != data_source_finitude_type::finite) {
        data_pipeline_ = {};

        throw_data_pipeline_error(std::move(maybe_current_example_), /*recoverable=*/true,
            "The data pipeline to yield from cannot be infinite.");
    }

    return maybe_current_example_.has_value();
}

data_pipeline
yield_from_data_source::invoke_function(data &example)
{
    return yield_fn_(example);
}

}  // namespace fairseq2n::detail
