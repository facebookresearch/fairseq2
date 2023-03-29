// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/tape.h"

namespace fairseq2 {

void
tape::record_data(const data &d)
{
    if (iter_ != storage_.end())
        throw std::logic_error{"New data can only be recorded to the end of the tape."};

    storage_.push_back(d);

    // At this point, the iterator is invalid because of the push_back call; do
    // not increment it.
    iter_ = storage_.end();
}

data
tape::read_data()
{
    if (iter_ == storage_.end())
        throw_corrupt();

    return *iter_++;
}

corrupt_tape_error::~corrupt_tape_error() = default;

}  // namespace fairseq2
