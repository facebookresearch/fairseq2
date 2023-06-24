// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class FAIRSEQ2_API str_splitter final : public data_processor {
public:
    explicit
    str_splitter(char sep = '\t') noexcept
      : sep_{sep}
    {}

    data
    operator()(const data &d) const override;

    data
    operator()(data &&d) const override;

private:
    char sep_;
};

}  // namespace fairseq2
