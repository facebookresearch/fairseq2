// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/list_data_source.h"

namespace fairseq2 {

list_data_source::list_data_source(const c10::IValue &v) noexcept
    : list_(v.toList()), pos_{list_.begin()}
{}

bool
list_data_source::move_next()
{
    if (pos_ == list_.end())
        return false;

    ++pos_;

    return true;
}

const c10::IValue &
list_data_source::current() const noexcept
{
    return *pos_;
}

void
list_data_source::reset()
{
    pos_ = list_.begin();
}

}  // namespace fairseq2
