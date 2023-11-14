// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/sentencepiece/sp_encoder.h"

#include <algorithm>
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
namespace {

std::int64_t
get_token_idx(const ImmutableSentencePieceText &spt, std::size_t idx) noexcept
{
    std::uint32_t id = spt.pieces(conditional_cast<int>(idx)).id();

    return static_cast<std::int64_t>(id);
}

}  // namespace
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

    ImmutableSentencePieceText spt{};

    if (opts_.enable_sampling())
        spt = model_->processor_->sample(d.as_string(), opts_.nbest_size(), opts_.alpha());
    else
        spt = model_->processor_->encode(d.as_string());

    const std::vector<std::string> &prefix_tokens = opts_.prefix_tokens();
    const std::vector<std::string> &suffix_tokens = opts_.suffix_tokens();

    std::size_t seq_len = spt.pieces_size() + prefix_tokens.size() + suffix_tokens.size();

    at::Tensor tensor = at::zeros({static_cast<std::int64_t>(seq_len)},
        at::dtype(at::kLong).device(at::kCPU).pinned_memory(opts_.pin_memory()));

    writable_memory_span tensor_bits = get_raw_mutable_storage(tensor);

    span tensor_data = cast<std::int64_t>(tensor_bits);

    if (opts_.reverse()) {
        std::size_t i = seq_len - 1;

        for (std::int64_t prefix_idx : prefix_token_indices_)
            tensor_data[i--] = prefix_idx;

        for (std::size_t j = 0; j < spt.pieces_size(); ++j)
            tensor_data[i--] = get_token_idx(spt, j);

        for (std::int64_t suffix_idx : suffix_token_indices_)
            tensor_data[i--] = suffix_idx;
    } else {
        std::size_t i = 0;

        for (std::int64_t prefix_idx : prefix_token_indices_)
            tensor_data[i++] = prefix_idx;

        for (std::size_t j = 0; j < spt.pieces_size(); ++j)
            tensor_data[i++] = get_token_idx(spt, j);

        for (std::int64_t suffix_idx : suffix_token_indices_)
            tensor_data[i++] = suffix_idx;
    }

    at::Device device = opts_.maybe_device().value_or(at::kCPU);
    if (device != at::kCPU)
        tensor = tensor.to(device);

    return tensor;
}

data
sp_encoder::encode_as_tokens(data &&d) const
{
    if (!d.is_string())
        throw_<std::invalid_argument>(
            "The input data must be of type `string`, but is of type `{}` instead.", d.type());

    ImmutableSentencePieceText spt{};

    if (opts_.enable_sampling())
        spt = model_->processor_->sample(d.as_string(), opts_.nbest_size(), opts_.alpha());
    else
        spt = model_->processor_->encode(d.as_string());

    std::vector<data> tokens{};

    const std::vector<std::string> &prefix_tokens = opts_.prefix_tokens();
    const std::vector<std::string> &suffix_tokens = opts_.suffix_tokens();

    tokens.reserve(spt.pieces_size() + prefix_tokens.size() + suffix_tokens.size());

    for (const std::string &token : prefix_tokens)
        tokens.emplace_back(token);

    for (const auto &sp : spt.pieces())
        tokens.emplace_back(sp.piece());

    for (const std::string &token : suffix_tokens)
        tokens.emplace_back(token);

    if (opts_.reverse())
        std::reverse(tokens.begin(), tokens.end());

    return tokens;
}

}  // namespace fairseq2n
