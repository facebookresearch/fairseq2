// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <vector>

#include <ATen/Tensor.h>

#include "fairseq2/native/data/text/dicttokenizer/dict_model.h"
#include "fairseq2/native/span.h"
#include "fairseq2/native/data/data.h"
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class FAIRSEQ2_API dict_encoder final : public data_processor {

public:
    explicit
    dict_encoder(const dict_model *model, std::int64_t max_seq_len);

    data
    operator()(data &&d) const override;

private:
    const dict_model *model_;
    const std::int64_t max_seq_len_;

    at::Tensor
    encode(span<data> sentences) const;
};

}  // namespace fairseq2
