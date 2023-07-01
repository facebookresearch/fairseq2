// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/processors/custom_data_processor.h"

#include "fairseq2/native/py.h"

namespace fairseq2 {

data
custom_data_processor::process(data &&d) const
{
    // See the note [Python Finalization].
    detail::throw_if_py_is_finalizing();

    return fn_(std::move(d));
}

}  // namespace fairseq2
