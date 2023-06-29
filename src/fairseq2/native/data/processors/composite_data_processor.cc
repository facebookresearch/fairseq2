// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/processors/composite_data_processor.h"

namespace fairseq2 {

data
composite_data_processor::process(data &&d) const
{
    for (const std::shared_ptr<const data_processor> &p : processors_)
        d = p->process(std::move(d));

    return std::move(d);
}

}  // namespace fairseq2
