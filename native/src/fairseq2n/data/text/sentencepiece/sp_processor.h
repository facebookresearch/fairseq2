// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <sentencepiece/src/sentencepiece_processor.h>

#include "fairseq2n/data/text/sentencepiece/sp_model.h"

namespace fairseq2n::detail {

class sp_processor {
public:
    static std::unique_ptr<sp_processor>
    from_serialized(std::string_view serialized);

private:
    sp_processor(std::unique_ptr<sentencepiece::ModelProto> &&proto);

public:
    explicit
    sp_processor(std::string_view model_pathname, sp_model_options &&opts);

    sentencepiece::ImmutableSentencePieceText
    encode(std::string_view text) const;

    sentencepiece::ImmutableSentencePieceText
    sample(std::string_view text, std::int32_t nbest_size, float alpha) const;

    std::string
    decode(const std::vector<std::string_view> &tokens) const;

    std::int32_t
    token_to_index(std::string_view token) const;

    std::string_view
    index_to_token(std::int32_t idx) const;

    std::string
    serialize() const;

public:
    std::int32_t unk_idx;
    std::int32_t bos_idx;
    std::int32_t eos_idx;
    std::int32_t pad_idx;

    std::size_t vocabulary_size;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> native_;
};

}  // namespace fairseq2n::detail
