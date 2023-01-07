// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include <ATen/Tensor.h>

#include "fairseq2/native/api.h"
#include "fairseq2/native/span.h"
#include "fairseq2/native/data/data.h"
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class sp_model;

class FAIRSEQ2_API sp_decoder final : public data_processor {
public:
    explicit
    sp_decoder(const sp_model *m) noexcept
        : model_{m}
    {}

    data
    operator()(data &&d) const override;

private:
    std::vector<data>
    decode(at::Tensor &&t) const;

private:
    const sp_model *model_;
};

}  // namespace fairseq2
