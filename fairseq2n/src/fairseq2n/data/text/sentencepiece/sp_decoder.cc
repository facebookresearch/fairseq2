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
namespace detail {

class sp_decoder_op {
public:
    explicit
    sp_decoder_op(const sp_decoder *decoder, const sp_processor *processor, at::Tensor &&tensor);

    immutable_string &&
    run() &&;

private:
    void
    decode();

    template <typename T>
    void
    decode();

private:
    const sp_decoder *decoder_;
    const sp_processor *processor_;
    at::Tensor tensor_;
    immutable_string sentence_{};
};

sp_decoder_op::sp_decoder_op(
    const sp_decoder *decoder, const sp_processor *processor, at::Tensor &&tensor)
  : decoder_{decoder}, processor_{processor}, tensor_{std::move(tensor)}
{}

immutable_string &&
sp_decoder_op::run() &&
{
    tensor_ = tensor_.to(at::kCPU);

    decode();

    return std::move(sentence_);
}

void
sp_decoder_op::decode()
{
    switch (tensor_.scalar_type()) {
    case at::ScalarType::Short:
        decode<std::int16_t>();
        break;

    case at::ScalarType::Int:
        decode<std::int32_t>();
        break;

    case at::ScalarType::Long:
        decode<std::int64_t>();
        break;

    default:
        throw_<not_supported_error>(
            "`sp_decoder` supports only `torch.int16`, `torch.int32`, and `torch.int64` data types.");
    }
}

template <typename T>
void
sp_decoder_op::decode()
{
    std::int64_t seq_len = tensor_.size(0);

    std::vector<std::string_view> tokens{};

    tokens.reserve(static_cast<std::size_t>(seq_len));

    auto tensor_data = tensor_.accessor<T, 1>();

    for (std::int64_t j = 0; j < seq_len; j++) {
        T token_idx = tensor_data[decoder_->reverse_ ? seq_len - 1 - j : j];

        auto token_idx_32bit = conditional_cast<std::int32_t>(token_idx);

        std::string_view token = processor_->index_to_token(token_idx_32bit);

        tokens.push_back(token);
    }

    sentence_ = processor_->decode(tokens);
}

}  // namespace detail

sp_decoder::sp_decoder(std::shared_ptr<const sp_model> model, bool reverse) noexcept
  : model_{std::move(model)}, reverse_{reverse}
{}

data
sp_decoder::operator()(data &&d) const
{
    if (!d.is_tensor())
        throw_<std::invalid_argument>(
            "The input data must be of type `torch.Tensor`, but is of type `{}` instead.", d.type());

    at::Tensor tensor = d.as_tensor();

    if (tensor.dim() != 1)
        throw_<std::invalid_argument>(
            "The input tensor must be one dimensional, but has {} dimension(s) instead.", tensor.dim());

    return decode(std::move(tensor));
}

immutable_string
sp_decoder::decode(at::Tensor &&tensor) const
{
    return sp_decoder_op{this, model_->processor_.get(), std::move(tensor)}.run();
}

}  // namespace fairseq2n
