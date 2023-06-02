// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/dicttokenizer/dict_model.h"

#include <string>
#include <stdexcept>

namespace fairseq2 {

dict_model::dict_model(std::vector<std::string>&& vocab)
{
    index_to_token_ = std::move(vocab);
    init_token_to_index();
    index_to_token_.insert(index_to_token_.begin(), symbols.begin(), symbols.end());
}

std::size_t
dict_model::vocab_size() const
{
    return index_to_token_.size();
}

std::string_view
dict_model::index_to_token(std::int64_t idx) const
{
    auto unsigned_idx = static_cast<std::size_t>(idx); // we need idx to be signed for pytorch compatibility
    if (unsigned_idx > index_to_token_.size())
        throw std::invalid_argument("Index out of range: " + std::to_string(unsigned_idx));

    return index_to_token_[unsigned_idx];
}

std::int64_t
dict_model::token_to_index(std::string_view token) const
{
    if (auto pos = token_to_index_.find(std::string(token)); pos != token_to_index_.end())
        return pos->second;

    return unk_token_idx;
}

void
dict_model::init_token_to_index()
{
    auto index = static_cast<std::int64_t>(symbols.size());
    for (const auto& word: index_to_token_) {
        if (token_to_index_.find(word) != token_to_index_.end())
            throw std::invalid_argument("vocab argument should contain unique words only. Found duplicate for: '" + word + "'.");

        token_to_index_.insert(std::pair{word, index++});
    }
}

} // fairseq2::detail
