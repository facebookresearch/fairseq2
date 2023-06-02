// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/dicttokenizer/dict_decoder.h"

#include <ATen/core/TensorBody.h>

namespace fairseq2 {

dict_decoder::dict_decoder(const dict_model *model) noexcept
    : model_{model}
{
}

data
dict_decoder::operator()(data &&d) const
{
    if (!d.is_tensor())
        throw std::invalid_argument{"Decoder expects as input a tensor."};

    at::Tensor tensor = d.as_tensor();
    if (tensor.dim() == 1)
        tensor = tensor.unsqueeze(0);

    return decode(std::move(tensor));
}

std::vector<data>
dict_decoder::decode(at::Tensor &&tensor) const
{
    tensor = tensor.to(at::kCPU);
    auto tensor_a = tensor.accessor<std::int64_t, 2>(); // TODO find a better way to access data
    auto batch_size = tensor.size(0);
    auto seq_dim = tensor.size(1);

    std::vector<data> sentences;
    sentences.reserve(static_cast<std::size_t>(batch_size));
    for (auto i = 0; i < batch_size; ++i) {
        std::string sentence;
        for (auto j = 0; j < seq_dim; ++j) {
            auto token = this->model_->index_to_token(tensor_a[i][j]);
            sentence.append(token);
            sentence.append(" ");
        }
        sentences.emplace_back(sentence.substr(0, sentence.length() - 1));
    }

    return sentences;
}

} // fairseq2
