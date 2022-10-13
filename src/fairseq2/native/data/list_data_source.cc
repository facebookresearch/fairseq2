// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/list_data_source.h"

namespace fairseq2 {

list_data_source::list_data_source(const ivalue &v) noexcept : list_(v.toList()), pos_{list_.end()}
{}

bool
list_data_source::move_next()
{
    if (iterating_) {
        if (pos_ == list_.end())
            return false;

        ++pos_;

        return true;
    } else {
        pos_ = list_.begin();

        iterating_ = true;

        // Check if we have an empty list.
        return pos_ != list_.end();
    }
}

ivalue
list_data_source::current() const noexcept
{
    return *pos_;
}

void
list_data_source::reset()
{
    pos_ = list_.end();

    iterating_ = false;
}

void
list_data_source::seek(std::ptrdiff_t offset, whence w)
{
    generic_list<ivalue>::iterator pos{};

    switch (w) {
    case whence::begin:
        pos = list_.begin();
        break;
    case whence::current:
        if (!iterating_)
            return;

        pos = pos_;
        break;
    case whence::end:
        pos = list_.end() - 1;
        break;
    }

    if (offset < 0)
        pos_ -= static_cast<std::size_t>(-offset);
    else
        pos_ += static_cast<std::size_t>(+offset);
}

bool
list_data_source::seekable() const noexcept
{
    return true;
}

}  // namespace fairseq2
