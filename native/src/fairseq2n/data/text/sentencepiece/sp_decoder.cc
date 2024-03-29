// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/sentencepiece/sp_decoder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <ATen/Functions.h>
#include <ATen/ScalarType.h>
#include <ATen/Storage.h>

#include "fairseq2n/exception.h"
#include "fairseq2n/fmt.h"
#include "fairseq2n/data/text/sentencepiece/sp_model.h"
#include "fairseq2n/data/text/sentencepiece/sp_processor.h"
#include "fairseq2n/detail/exception.h"
#include "fairseq2n/utils/cast.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

sp_decoder::sp_decoder(std::shared_ptr<const sp_model> model, bool reverse) noexcept
  : model_{std::move(model)}, reverse_{reverse}
{}

data
sp_decoder::operator()(data &&d) const
{
    if (!d.is_tensor())
        throw_<std::invalid_argument>(
            "The input data must be of type `torch.Tensor`, but is of type `{}` instead.", d.type());

    return decode(d.as_tensor());
}

std::string
sp_decoder::decode(const at::Tensor &tensor) const
{
    if (tensor.dim() != 1)
        throw_<std::invalid_argument>(
            "The input tensor must be one dimensional, but has {} dimension(s) instead.", tensor.dim());

    at::Tensor cpu_tensor = tensor.to(at::kCPU);

    switch (tensor.scalar_type()) {
    case at::ScalarType::Short:
        return do_decode<std::int16_t>(cpu_tensor);

    case at::ScalarType::Int:
        return do_decode<std::int32_t>(cpu_tensor);

    case at::ScalarType::Long:
        return do_decode<std::int64_t>(cpu_tensor);

    default:
        throw_<not_supported_error>(
            "`sp_decoder` supports only `torch.int16`, `torch.int32`, and `torch.int64` data types.");
    }
}

std::string
sp_decoder::decode_from_tokens(const std::vector<std::string_view> &tokens) const
{
    if (reverse_) {
        auto tokens_reversed = tokens;

        std::reverse(tokens_reversed.begin(), tokens_reversed.end());

        return model_->processor_->decode(tokens_reversed);
    } else
        return model_->processor_->decode(tokens);
}

template <typename T>
std::string
sp_decoder::do_decode(const at::Tensor &tensor) const
{
    std::int64_t seq_len = tensor.size(0);

    std::vector<std::string_view> tokens{};

    tokens.reserve(static_cast<std::size_t>(seq_len));

    auto tensor_data = tensor.accessor<T, 1>();

    for (std::int64_t j = 0; j < seq_len; j++) {
        T token_idx = tensor_data[reverse_ ? seq_len - 1 - j : j];

        auto token_idx_32bit = conditional_cast<std::int32_t>(token_idx);

        std::string_view token = model_->processor_->index_to_token(token_idx_32bit);

        tokens.push_back(token);
    }

    return model_->processor_->decode(tokens);
}

}  // namespace fairseq2n
