// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/filter_data_source.h"

namespace fairseq2::detail {

std::optional<data>
filter_data_source::next()
{
    std::optional<data> d{};

    while ((d = inner_->next()))
        if (invoke_function(*d))
            break;

    return d;
}

void
filter_data_source::reset()
{
    inner_->reset();
}

void
filter_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
filter_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

bool
filter_data_source::invoke_function(data &d)
{
    try {
        return predicate_fn_(d);
    } catch (const data_pipeline_error &) {
        throw;
    } catch (...) {
        data_pipeline_error::throw_nested(
            "The filter operation has failed.", std::move(d));
    }
}

} // fairseq2::detail
