// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/constant_data_source.h"

#include "fairseq2n/data/data.h"

namespace fairseq2n::detail {

std::optional<data>
constant_data_source::next()
{
    if (maybe_key_)
        return data_dict{{*maybe_key_, example_}};

    return example_;
}

void
constant_data_source::reset(bool)
{}

void
constant_data_source::record_position(tape &, bool) const
{}

void
constant_data_source::reload_position(tape &, bool)
{}

data_source_finitude_type
constant_data_source::finitude_type() const noexcept
{
    return data_source_finitude_type::pseudo_infinite;
}

}
