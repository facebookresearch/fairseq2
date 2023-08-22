// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/sentencepiece/sp_model.h"

#include "fairseq2n/data/text/sentencepiece/sp_processor.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

sp_model
sp_model::from_serialized(std::string_view serialized)
{
    std::unique_ptr<sp_processor> processor = sp_processor::from_serialized(serialized);

    return sp_model{std::move(processor)};
}

sp_model::sp_model(std::unique_ptr<sp_processor> &&processor) noexcept
  : processor_{std::move(processor)}
{}

sp_model::sp_model(std::string_view pathname, sp_model_options opts)
{
    processor_ = std::make_unique<sp_processor>(pathname, std::move(opts));
}

sp_model::sp_model(sp_model &&) noexcept = default;
sp_model &sp_model::operator=(sp_model &&) noexcept = default;

sp_model::~sp_model() = default;

std::int64_t
sp_model::token_to_index(std::string_view token) const
{
    return processor_->token_to_index(token);
}

std::string_view
sp_model::index_to_token(std::int64_t idx) const
{
    return processor_->index_to_token(static_cast<std::int32_t>(idx));
}

std::optional<std::int64_t>
sp_model::unk_idx() const
{
    if (processor_->unk_idx < 0)
        return std::nullopt;

    return processor_->unk_idx;
}

std::optional<std::int64_t>
sp_model::bos_idx() const
{
    if (processor_->bos_idx < 0)
        return std::nullopt;

    return processor_->bos_idx;
}

std::optional<std::int64_t>
sp_model::eos_idx() const
{
    if (processor_->eos_idx < 0)
        return std::nullopt;

    return processor_->eos_idx;
}

std::optional<std::int64_t>
sp_model::pad_idx() const
{
    if (processor_->pad_idx < 0)
        return std::nullopt;

    return processor_->pad_idx;
}

std::size_t
sp_model::vocabulary_size() const
{
    return processor_->vocabulary_size;
}

std::string
sp_model::serialize() const
{
    return processor_->serialize();
}

}  // namespace fairseq2n
