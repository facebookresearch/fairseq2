// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/dict_tokenizer/dict_encoder.h"

#include "fairseq2/native/span.h"
#include "fairseq2/native/data/immutable_string.h"

#include <ATen/ops/full.h>

namespace fairseq2 {

dict_encoder::dict_encoder(const dict_model *model, std::int64_t max_seq_len)
    : model_{model}, max_seq_len_{max_seq_len}
{
}

data
dict_encoder::process(data &&d) const
{
    if (!d.is_string())
        throw std::invalid_argument{"The input data must be of type string."};

    return encode(d.as_string());
}

at::Tensor
dict_encoder::encode(const immutable_string &sentence) const
{
    // TODO compute seq_len for each batch instead of using fixed max_seq_len

    auto tensor = at::full({max_seq_len_}, this->model_->pad_token_idx, at::TensorOptions().dtype(at::kLong));
    auto tensor_a = tensor.accessor<std::int64_t, 1>();

    auto tokens = sentence.split(' ');
    auto token_count = std::min(max_seq_len_ - 2, static_cast<std::int64_t>(tokens.size()));

    tensor_a[0] = this->model_->bos_token_idx;
    for (auto j = 0; j < token_count; ++j)
        tensor_a[j + 1] = this->model_->token_to_index(tokens[static_cast<std::size_t>(j)]);

    tensor_a[token_count + 1] = this->model_->eos_token_idx;

    return tensor;
}

} // namespace fairseq2
