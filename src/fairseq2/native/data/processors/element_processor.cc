// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/processors/element_processor.h"

namespace fairseq2 {

data
element_processor::process(data &&d) const
{
    selector_.visit(d, [this](data &e) {
        e = processor_->process(std::move(e));
    });

    return std::move(d);
}

}  // namespace fairseq2
