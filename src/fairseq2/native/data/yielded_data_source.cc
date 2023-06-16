// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/yielded_data_source.h"

namespace fairseq2::detail {

std::optional<data>
yielded_data_source::next()
{
    std::optional<data> d{};

    while (!(d = data_pipeline_.next())) {
        if (!load_data_pipeline(inner_->next()))
            break;
    }

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
        try {
            data_pipeline_ = fn_(*example_);
        } catch (const data_pipeline_error &) {
            throw;
        } catch (...) {
            data_pipeline_error::throw_nested(
                "The yield function has failed.", std::move(example_));
        }
    }
    else
        data_pipeline_ = {};

    data_pipeline_.reload_position(t);

    inner_->reload_position(t);
}

bool
yielded_data_source::load_data_pipeline(std::optional<data> &&d)
{
    example_ = std::move(d);

    if (example_) {
        try {
            data_pipeline_ = fn_(*example_);
        } catch (const data_pipeline_error &) {
            throw;
        } catch (...) {
            data_pipeline_error::throw_nested(
                "The yield function has failed.", std::move(example_));
        }
    } else
        data_pipeline_ = {};

    return example_.has_value();
}

}  // namespace fairseq2::detail
