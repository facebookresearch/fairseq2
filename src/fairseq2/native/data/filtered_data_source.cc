// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/filtered_data_source.h"

namespace fairseq2::detail {

std::optional<data>
filtered_data_source::next()
{
    auto result = inner_->next();
    while (result.has_value() && !try_predicate(result.value()))
        result = inner_->next();

    return result;
}

void
filtered_data_source::reset()
{
    inner_->reset();
}

void
filtered_data_source::record_position(tape &t) const
{
    inner_->record_position(t);
}

void
filtered_data_source::reload_position(tape &t)
{
    inner_->reload_position(t);
}

bool
filtered_data_source::try_predicate(data &value)
{
    try {
        return predicate_(value);
    }
    catch (const data_pipeline_error &) {
        throw;
    }
    catch (...) {
        data_pipeline_error::throw_nested(
            "The predicate function has failed.", std::move(value));
    }
}

} // fairseq2::detail
