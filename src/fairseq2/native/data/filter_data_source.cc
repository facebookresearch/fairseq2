// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/filter_data_source.h"

#include <exception>

#include "fairseq2/native/data/detail/exception.h"

namespace fairseq2::detail {

std::optional<data>
filter_data_source::next()
{
    std::optional<data> maybe_example{};

    while ((maybe_example = inner_->next()))
        if (invoke_function(*maybe_example))
            break;

    return maybe_example;
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
filter_data_source::invoke_function(data &example)
{
    try {
        return predicate_fn_(example);
    } catch (const data_pipeline_error &) {
        throw;
    } catch (const std::exception &) {
        throw_data_pipeline_error_with_nested(std::move(example),
            "The filter operation has failed. See nested exception for details.");
    }
}

} // fairseq2::detail
