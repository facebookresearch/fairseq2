// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include <ATen/Tensor.h>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/text/dict_tokenizer/dict_model.h"
#include "fairseq2/native/span.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2 {

class FAIRSEQ2_API dict_decoder final {

public:
    explicit
    dict_decoder(const dict_model *model) noexcept;

    data
    operator()(data &&d) const;

private:
    const dict_model *model_;

    std::vector<data>
    decode(at::Tensor &&t) const;
};

}  // namespace fairseq2
