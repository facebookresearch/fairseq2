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
    if (key_)
        return data_dict{{*key_, example_}};

    return example_;
}

void
constant_data_source::reset()
{}

void
constant_data_source::record_position(tape &) const
{}

void
constant_data_source::reload_position(tape &)
{}

bool
constant_data_source::is_infinite() const noexcept
{
    return true;
}

}
