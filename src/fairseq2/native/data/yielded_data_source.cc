// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/yielded_data_source.h"

#include "fairseq2/native/py.h"

namespace fairseq2::detail {

std::optional<data>
yielded_data_source::next()
{
    std::optional<data> d{};

    while (!(d = data_pipeline_.next()) && load_next_data_pipeline());

    return d;
}

void
yielded_data_source::reset()
{
    example_ = {};

    data_pipeline_ = {};

    inner_->reset();
}

void
yielded_data_source::record_position(tape &t) const
{
    t.record(example_);

    if (example_)
        data_pipeline_.record_position(t);

    inner_->record_position(t);
}

void
yielded_data_source::reload_position(tape &t)
{
    example_ = t.read<std::optional<data>>();

    if (example_) {
        data_pipeline_ = invoke_fn(*example_);

        data_pipeline_.reload_position(t);
    }
    else
        data_pipeline_ = {};

    inner_->reload_position(t);
}

bool
yielded_data_source::load_next_data_pipeline()
{
    // See the note [Python Finalization].
    if (py_is_finalizing())
        return false;

    example_ = inner_->next();

    if (example_)
        data_pipeline_ = invoke_fn(*example_);
    else
        data_pipeline_ = {};

    return example_.has_value();
}

data_pipeline
yielded_data_source::invoke_fn(data &d)
{
    try {
        return fn_(d);
    } catch (const data_pipeline_error &) {
        throw;
    } catch (...) {
        data_pipeline_error::throw_nested(
            "The yield operation has failed.", std::move(d));
    }
}

}  // namespace fairseq2::detail
