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
filter_data_source::reset(bool reset_rng)
{
    inner_->reset(reset_rng);
}

void
filter_data_source::record_position(tape &t, bool strict) const
{
    inner_->record_position(t, strict);
}

void
filter_data_source::reload_position(tape &t, bool strict)
{
    inner_->reload_position(t, strict);
}

data_source_finitude_type
filter_data_source::finitude_type() const noexcept
{
    return inner_->finitude_type();
}

bool
filter_data_source::invoke_function(data &example)
{
    return predicate_fn_(example);
}

} // fairseq2n::detail
