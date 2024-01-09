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
yield_from_data_source::reset()
{
    maybe_current_example_ = {};

    data_pipeline_ = {};

    inner_->reset();
}

void
yield_from_data_source::record_position(tape &t) const
{
    t.record(maybe_current_example_);

    if (maybe_current_example_)
        data_pipeline_.record_position(t);

    inner_->record_position(t);
}

void
yield_from_data_source::reload_position(tape &t)
{
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

    inner_->reload_position(t);
}

bool
yield_from_data_source::is_infinite() const noexcept
{
    return inner_->is_infinite();
}

bool
yield_from_data_source::load_next_data_pipeline()
{
    maybe_current_example_ = inner_->next();

    if (maybe_current_example_)
        data_pipeline_ = invoke_function(*maybe_current_example_);
    else
        data_pipeline_ = {};

    if (data_pipeline_.is_infinite()) {
        data_pipeline_ = {};

        throw_data_pipeline_error(std::move(maybe_current_example_), /*recoverable=*/true,
            "The data pipeline to yield from cannot be infinite.");
    }

    return maybe_current_example_.has_value();
}

data_pipeline
yield_from_data_source::invoke_function(data &example)
{
    try {
        return yield_fn_(example);
    } catch (const data_pipeline_error &) {
        throw;
    } catch (const std::exception &) {
        throw_data_pipeline_error_with_nested(std::move(example), /*recoverable=*/true,
            "The yield operation has failed. See nested exception for details.");
    }
}

}  // namespace fairseq2n::detail
