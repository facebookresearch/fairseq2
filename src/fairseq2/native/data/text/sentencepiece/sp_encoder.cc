// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//#define _GLIBCXX_USE_TBB_PAR_BACKEND 1

#include "fairseq2/native/data/text/sentencepiece/sp_encoder.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ATen/Functions.h>
#include <ATen/Storage.h>
#include <oneapi/tbb.h>

#include "fairseq2/native/memory.h"
#include "fairseq2/native/exception.h"
#include "fairseq2/native/data/text/sentencepiece/sp_model.h"
#include "fairseq2/native/data/text/sentencepiece/sp_processor.h"
#include "fairseq2/native/utils/cast.h"

using sentencepiece::ImmutableSentencePieceText;

namespace fairseq2 {
namespace detail {

class encoder_op {
public:
    explicit
    encoder_op(const sp_encoder *e, const sp_processor *proc, span<const data> sentences);

    at::Tensor &&
    run() &&;

private:
    void
    encode_strings();

    void
    find_longest_sequence() noexcept;

    void
    compute_sequence_dimension() noexcept;

    void
    compute_batch_size() noexcept;

    void
    init_tensor();

    void
    fill_tensor();

    template <typename T>
    void
    fill_tensor();

private:
    const sp_encoder *encoder_;
    const sp_processor *processor_;
    span<const data> sentences_;
    std::vector<ImmutableSentencePieceText> spts_;
    std::size_t extra_tokens_len_{};
    std::int64_t max_seq_len_{};
    std::int64_t seq_dim_{};
    std::int64_t batch_size_{};
    at::Tensor tensor_{};
};

}  // namespace detail

sp_encoder::sp_encoder(std::shared_ptr<const sp_model> m, sp_encoder_options opts)
    : model_{std::move(m)}, opts_{std::move(opts)}
{
    if (!at::isIntegralType(opts_.dtype(), /*includeBool=*/false))
        throw std::invalid_argument{"The output data type must be integral."};

    prefix_token_indices_.reserve(opts_.prefix_tokens().size());
    suffix_token_indices_.reserve(opts_.suffix_tokens().size());

    for (const std::string &token : opts_.prefix_tokens())
        prefix_token_indices_.push_back(model_->token_to_index(token));

    for (const std::string &token : opts_.suffix_tokens())
        suffix_token_indices_.push_back(model_->token_to_index(token));
}

data
sp_encoder::operator()(const data &d) const
{
    if (d.is_list())
        return encode(d.as_list());
    else if (d.is_string()) {
        at::Tensor t = encode(as_singleton_span(d));

        return t.squeeze(0);
    } else
        throw std::invalid_argument{
            "The SentencePiece encoder expects as input a string or a list of strings."};
}

data
sp_encoder::operator()(data &&d) const
{
    return (*this)(d);
}

at::Tensor
sp_encoder::encode(span<const data> sentences) const
{
    detail::encoder_op op{this, model_->processor_.get(), sentences};

    return std::move(op).run();
}

namespace detail {

template <typename T>
T
get_token_idx(const ImmutableSentencePieceText &spt, std::size_t idx) noexcept
{
    std::uint32_t id = spt.pieces(conditional_cast<int>(idx)).id();

    return static_cast<T>(id);
}

encoder_op::encoder_op(const sp_encoder *e, const sp_processor *proc, span<const data> sentences)
    : encoder_{e}, processor_{proc}, sentences_{sentences}, spts_(sentences_.size())
{
    extra_tokens_len_ += encoder_->prefix_token_indices_.size();
    extra_tokens_len_ += encoder_->suffix_token_indices_.size();
}

at::Tensor &&
encoder_op::run() &&
{
    encode_strings();

    find_longest_sequence();

    compute_sequence_dimension();

    compute_batch_size();

    init_tensor();

    fill_tensor();

    std::optional<at::Device> device = encoder_->opts_.device();
    if (device)
        tensor_ = tensor_.to(*device);

    return std::move(tensor_);
}

void
encoder_op::encode_strings()
{
    auto op = [this](const tbb::blocked_range<std::size_t> &rng) {
        auto &opts = encoder_->opts_;

        for (auto i = rng.begin(); i < rng.end(); ++i) {
            const data &d = sentences_[i];

            if (!d.is_string()) {
                throw std::invalid_argument{
                    "The SentencePiece encoder expects all elements of the input to be strings."};
            }

            if (opts.enable_sampling())
                spts_[i] = processor_->sample(d.as_string(), opts.nbest_size(), opts.alpha());
            else
                spts_[i] = processor_->encode(d.as_string());
        }
    };

    tbb::blocked_range<std::size_t> full_rng{0, sentences_.size()};

    if (encoder_->opts_.disable_parallelism())
        op(full_rng);
    else
        tbb::parallel_for(full_rng, op);
}

inline void
encoder_op::find_longest_sequence() noexcept
{
    auto iter = std::max_element(spts_.begin(), spts_.end(), [](const auto &a, const auto &b) {
        return a.pieces_size() < b.pieces_size();
    });

    max_seq_len_ = static_cast<std::int64_t>(iter->pieces_size() + extra_tokens_len_);
}

inline void
encoder_op::compute_sequence_dimension() noexcept
{
    auto &opts = encoder_->opts_;

    std::int64_t seq_dim = std::max(max_seq_len_, opts.pad_to_length().value_or(0));

    auto r = seq_dim % opts.pad_to_multiple();
    if (r == 0)
        seq_dim_ = seq_dim;
    else
        seq_dim_ = seq_dim - r + opts.pad_to_multiple();
}

inline void
encoder_op::compute_batch_size() noexcept
{
    auto batch_size = static_cast<std::int64_t>(spts_.size());

    batch_size_ = std::max(batch_size, encoder_->opts_.batch_size().value_or(0));
}

void
encoder_op::init_tensor()
{
    auto &opts = encoder_->opts_;

    tensor_ = at::full({batch_size_, seq_dim_}, encoder_->model_->pad_idx(),
        at::dtype(opts.dtype()).device(at::kCPU).pinned_memory(opts.pin_memory()));
}

void
encoder_op::fill_tensor()
{
    switch (encoder_->opts_.dtype()) {
    case at::ScalarType::Short:
        fill_tensor<std::int16_t>();
        break;

    case at::ScalarType::Int:
        fill_tensor<std::int32_t>();
        break;

    case at::ScalarType::Long:
        fill_tensor<std::int64_t>();
        break;

    default:
        throw not_supported_error{
            "The specified integral type is not supported."};
    }
}

template <typename T>
void
encoder_op::fill_tensor()
{
    auto seq_dim = static_cast<std::size_t>(seq_dim_);

    const at::Storage &s = tensor_.storage();

    writable_memory_span bits{s.unsafe_data<std::byte>(), s.nbytes()};

    span data = cast<T>(bits);

    auto op = [this, data, seq_dim](const tbb::blocked_range<std::size_t> &rng) {
        auto &opts = encoder_->opts_;

        for (auto i = rng.begin(); i < rng.end(); ++i) {
            auto &spt = spts_[i];

            std::size_t seq_len = spt.pieces_size() + extra_tokens_len_;

            span seq_data = data.subspan(
                i * seq_dim + (opts.left_pad() ? seq_dim - seq_len : 0), seq_len);

            if (opts.reverse()) {
                std::size_t j = seq_len - 1;

                for (std::int32_t prefix_idx : encoder_->prefix_token_indices_)
                    seq_data[j--] = conditional_cast<T>(prefix_idx);

                for (std::size_t k = 0; k < spt.pieces_size(); k++)
                    seq_data[j--] = get_token_idx<T>(spt, k);

                for (std::int32_t suffix_idx : encoder_->suffix_token_indices_)
                    seq_data[j--] = conditional_cast<T>(suffix_idx);
            } else {
                std::size_t j = 0;

                for (std::int32_t prefix_idx : encoder_->prefix_token_indices_)
                    seq_data[j++] = conditional_cast<T>(prefix_idx);

                for (std::size_t k = 0; k < spt.pieces_size(); k++)
                    seq_data[j++] = get_token_idx<T>(spt, k);

                for (std::int32_t suffix_idx : encoder_->suffix_token_indices_)
                    seq_data[j++] = conditional_cast<T>(suffix_idx);
            }
        }
    };

    tbb::blocked_range<std::size_t> full_rng{0, spts_.size()};

    if (encoder_->opts_.disable_parallelism())
        op(full_rng);
    else
        tbb::parallel_for(full_rng, op);
}

}  // namespace detail
}  // namespace fairseq2
