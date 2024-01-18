// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/filter_data_source.h"

#include <exception>

#include "fairseq2n/data/detail/exception.h"

namespace fairseq2n::detail {

std::optional<data>
filter_data_source::next()
{
    while (std::optional<data> maybe_example = inner_->next())
        if (invoke_function(*maybe_example))
            return maybe_example;

    return std::nullopt;
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
filter_data_source::is_infinite() const noexcept
{
    return inner_->is_infinite();
}

bool
filter_data_source::invoke_function(data &example)
{
    try {
        return predicate_fn_(example);
    } catch (const data_pipeline_error &) {
        throw;
    } catch (const std::exception &) {
        throw_data_pipeline_error_with_nested(std::move(example), /*recoverable=*/true,
            "The filter operation has failed. See nested exception for details.");
    }
}

} // fairseq2n::detail
