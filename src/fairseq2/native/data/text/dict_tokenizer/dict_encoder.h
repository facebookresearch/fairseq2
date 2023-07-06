// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <vector>

#include <ATen/Tensor.h>

#include "fairseq2/native/data/text/dict_tokenizer/dict_model.h"
#include "fairseq2/native/span.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2 {

class immutable_string;

class FAIRSEQ2_API dict_encoder final {

public:
    explicit
    dict_encoder(const dict_model *model, std::int64_t max_seq_len);

    data
    operator()(data &&d) const;

private:
    const dict_model *model_;
    const std::int64_t max_seq_len_;

    at::Tensor
    encode(const immutable_string &sentence) const;
};

}  // namespace fairseq2
