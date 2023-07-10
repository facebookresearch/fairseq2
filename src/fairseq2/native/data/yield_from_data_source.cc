// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/yield_from_data_source.h"

#include <exception>

#include "fairseq2/native/data/detail/exception.h"

namespace fairseq2::detail {

std::optional<data>
yield_from_data_source::next()
{
    std::optional<data> d{};

    while (!(d = data_pipeline_.next()))
        if (!load_next_data_pipeline())
            break;

    return d;
}

void
yield_from_data_source::reset()
{
    example_ = {};

    data_pipeline_ = {};

    inner_->reset();
}

void
yield_from_data_source::record_position(tape &t) const
{
    t.record(example_);

    if (example_)
        data_pipeline_.record_position(t);

    inner_->record_position(t);
}

void
yield_from_data_source::reload_position(tape &t)
{
    example_ = t.read<std::optional<data>>();

    if (example_) {
        // The assumption we make is that the recorded example will fully
        // reconstruct the original yielded-from data pipeline.
        data_pipeline_ = invoke_function(*example_);

        // With that assumption, we restore the recorded state.
        data_pipeline_.reload_position(t);
    }
    else
        data_pipeline_ = {};

    inner_->reload_position(t);
}

bool
yield_from_data_source::load_next_data_pipeline()
{
    example_ = inner_->next();

    if (example_)
        data_pipeline_ = invoke_function(*example_);
    else
        data_pipeline_ = {};

    return example_.has_value();
}

data_pipeline
yield_from_data_source::invoke_function(data &d)
{
    try {
        return yield_fn_(d);
    } catch (const data_pipeline_error &) {
        throw;
    } catch (const std::exception &) {
        throw_data_pipeline_error_with_nested(std::move(d),
            "The yield operation has failed. See nested exception for details.");
    }
}

}  // namespace fairseq2::detail
