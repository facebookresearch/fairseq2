// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/dicttokenizer/dict_encoder.h"

#include "fairseq2/native/span.h"
#include <ATen/ops/full.h>

namespace fairseq2 {

dict_encoder::dict_encoder(const dict_model *model, std::int64_t max_seq_len)
    : model_{model}, max_seq_len_{max_seq_len}
{
}

data
dict_encoder::operator()(const data &d) const
{
    if (d.is_list())
        return encode(d.as_list());
    else
        throw std::invalid_argument{"Encoder expects as input a list of strings."};
}

data
dict_encoder::operator()(data &&d) const
{
    return (*this)(d);
}

at::Tensor
dict_encoder::encode(span<const data> sentences) const
{
    // TODO compute seq_len for each batch instead of using fixed max_seq_len

    auto batch_size = static_cast<std::int64_t>(sentences.size());
    auto tensor = at::full({batch_size, max_seq_len_}, this->model_->pad_token_idx, at::TensorOptions().dtype(at::kLong));
    auto tensor_a = tensor.accessor<std::int64_t, 2>();

    for (auto i = 0; i < batch_size; ++i) {
        auto tokens = sentences[static_cast<std::size_t>(i)].as_string().split(' ');
        auto token_count = std::min(max_seq_len_ - 2, static_cast<std::int64_t>(tokens.size()));

        tensor_a[i][0] = this->model_->bos_token_idx;
        for (auto j = 0; j < token_count; ++j)
            tensor_a[i][j + 1] = this->model_->token_to_index(tokens[static_cast<std::size_t>(j)]);

        tensor_a[i][token_count + 1] = this->model_->eos_token_idx;
    }

    return tensor;
}

} // namespace fairseq2
