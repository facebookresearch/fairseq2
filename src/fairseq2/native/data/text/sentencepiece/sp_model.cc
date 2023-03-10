// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/sentencepiece/sp_model.h"

#include "fairseq2/native/data/text/sentencepiece/sp_processor.h"

using namespace fairseq2::detail;

namespace fairseq2 {

sp_model::sp_model(std::string_view pathname, sp_model_options opts)
{
    processor_ = std::make_unique<sp_processor>(pathname, std::move(opts));
}

sp_model
sp_model::from_serialized(std::string_view serialized)
{
    std::unique_ptr<sp_processor> proc = sp_processor::from_serialized(serialized);

    return sp_model{std::move(proc)};
}

sp_model::sp_model(sp_model &&) noexcept = default;
sp_model &sp_model::operator=(sp_model &&) noexcept = default;

sp_model::~sp_model() = default;

std::int32_t
sp_model::token_to_index(std::string_view token) const
{
    return processor_->token_to_index(token);
}

std::string_view
sp_model::index_to_token(std::int32_t idx) const
{
    return processor_->index_to_token(idx);
}

std::int32_t
sp_model::unk_idx() const
{
    return processor_->unk_idx;
}

std::int32_t
sp_model::bos_idx() const
{
    return processor_->bos_idx;
}

std::int32_t
sp_model::eos_idx() const
{
    return processor_->eos_idx;
}

std::int32_t
sp_model::pad_idx() const
{
    return processor_->pad_idx;
}

std::size_t
sp_model::vocab_size() const
{
    return processor_->vocab_size;
}

std::string
sp_model::serialize() const
{
    return processor_->serialize();
}

sp_model::sp_model(std::unique_ptr<sp_processor> &&proc) noexcept
    : processor_{std::move(proc)}
{}

}  // namespace fairseq2
