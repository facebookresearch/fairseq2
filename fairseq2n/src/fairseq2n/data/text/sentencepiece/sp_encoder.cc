// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/sentencepiece/sp_encoder.h"

#include <cstddef>
#include <stdexcept>

#include <ATen/Functions.h>

#include "fairseq2n/fmt.h"
#include "fairseq2n/memory.h"
#include "fairseq2n/span.h"
#include "fairseq2n/data/immutable_string.h"
#include "fairseq2n/data/detail/tensor_helpers.h"
#include "fairseq2n/data/text/sentencepiece/sp_model.h"
#include "fairseq2n/data/text/sentencepiece/sp_processor.h"
#include "fairseq2n/detail/exception.h"
#include "fairseq2n/utils/cast.h"

using sentencepiece::ImmutableSentencePieceText;

using namespace fairseq2n::detail;

namespace fairseq2n {
namespace detail {

class sp_encoder_op {
public:
    explicit
    sp_encoder_op(
        const sp_encoder *encoder, const sp_processor *processor, immutable_string &&sentence);

    at::Tensor &&
    run() &&;

private:
    void
    encode_string();

private:
    const sp_encoder *encoder_;
    const sp_processor *processor_;
    immutable_string sentence_;
    ImmutableSentencePieceText spt_{};
    std::size_t extra_tokens_len_{};
    std::size_t seq_len_{};
    at::Tensor tensor_{};
};

namespace {

std::int64_t
get_token_idx(const ImmutableSentencePieceText &spt, std::size_t idx) noexcept
{
    std::uint32_t id = spt.pieces(conditional_cast<int>(idx)).id();

    return static_cast<std::int64_t>(id);
}

}  // namespace

sp_encoder_op::sp_encoder_op(
    const sp_encoder *encoder, const sp_processor *processor, immutable_string &&sentence)
  : encoder_{encoder}, processor_{processor}, sentence_{std::move(sentence)}
{
    extra_tokens_len_ += encoder_->prefix_token_indices_.size();
    extra_tokens_len_ += encoder_->suffix_token_indices_.size();
}

at::Tensor &&
sp_encoder_op::run() &&
{
    encode_string();

    tensor_ = at::zeros({static_cast<std::int64_t>(seq_len_)},
        at::dtype(at::kLong).device(at::kCPU).pinned_memory(encoder_->opts_.pin_memory()));

    writable_memory_span tensor_bits = get_raw_mutable_storage(tensor_);

    span tensor_data = cast<std::int64_t>(tensor_bits);

    if (encoder_->opts_.reverse()) {
        std::size_t i = seq_len_ - 1;

        for (std::int64_t prefix_idx : encoder_->prefix_token_indices_)
            tensor_data[i--] = prefix_idx;

        for (std::size_t j = 0; j < spt_.pieces_size(); ++j)
            tensor_data[i--] = get_token_idx(spt_, j);

        for (std::int64_t suffix_idx : encoder_->suffix_token_indices_)
            tensor_data[i--] = suffix_idx;
    } else {
        std::size_t i = 0;

        for (std::int64_t prefix_idx : encoder_->prefix_token_indices_)
            tensor_data[i++] = prefix_idx;

        for (std::size_t j = 0; j < spt_.pieces_size(); ++j)
            tensor_data[i++] = get_token_idx(spt_, j);

        for (std::int64_t suffix_idx : encoder_->suffix_token_indices_)
            tensor_data[i++] = suffix_idx;
    }

    at::Device device = encoder_->opts_.maybe_device().value_or(at::kCPU);
    if (device != at::kCPU)
        tensor_ = tensor_.to(device);

    return std::move(tensor_);
}

void
sp_encoder_op::encode_string()
{
    auto &opts = encoder_->opts_;

    if (opts.enable_sampling())
        spt_ = processor_->sample(sentence_, opts.nbest_size(), opts.alpha());
    else
        spt_ = processor_->encode(sentence_);

    seq_len_ = spt_.pieces_size() + extra_tokens_len_;
}

}  // namespace detail

sp_encoder::sp_encoder(std::shared_ptr<const sp_model> model, sp_encoder_options opts)
  : model_{std::move(model)}, opts_{std::move(opts)}
{
    // Initialize prefix/suffix indices.
    prefix_token_indices_.reserve(opts_.prefix_tokens().size());
    suffix_token_indices_.reserve(opts_.suffix_tokens().size());

    for (const std::string &token : opts_.prefix_tokens())
        prefix_token_indices_.push_back(model_->token_to_index(token));

    for (const std::string &token : opts_.suffix_tokens())
        suffix_token_indices_.push_back(model_->token_to_index(token));

    // Create prefix/suffix index tensors.
    auto make_index_tensor = [this](const std::vector<std::int64_t> &indices)
    {
        at::Tensor tensor = at::empty(
            conditional_cast<std::int64_t>(indices.size()), at::dtype(at::kLong).device(at::kCPU));

        auto tensor_data = tensor.accessor<std::int64_t, 1>();

        std::int64_t i = 0;
        for (std::int64_t idx : indices)
            tensor_data[i++] = idx;

        at::Device device = opts_.maybe_device().value_or(at::kCPU);
        if (device != at::kCPU)
            tensor = tensor.to(device);

        return tensor;
    };

    if (!prefix_token_indices_.empty())
        prefix_index_tensor_ = make_index_tensor(prefix_token_indices_);

    if (!suffix_token_indices_.empty())
        suffix_index_tensor_ = make_index_tensor(suffix_token_indices_);
}

data
sp_encoder::operator()(data &&d) const
{
    if (!d.is_string())
        throw_<std::invalid_argument>(
            "The input data must be of type `string`, but is of type `{}` instead.", d.type());

    return encode(std::move(d).as_string());
}

at::Tensor
sp_encoder::encode(immutable_string &&sentence) const
{
    return sp_encoder_op{this, model_->processor_.get(), std::move(sentence)}.run();
}

}  // namespace fairseq2n
