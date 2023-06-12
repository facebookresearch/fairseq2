// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "fairseq2/native/api.h"

namespace fairseq2 {

class FAIRSEQ2_API dict_model {

public:
    explicit
    dict_model(std::vector<std::string>&& vocab, bool insert_symbols = true);

    std::int64_t
    token_to_index(std::string_view token) const;

    std::string_view
    index_to_token(std::int64_t idx) const;

    const std::vector<std::string>&
    vocab() const;

    std::size_t vocab_size() const;

    const std::int64_t unk_token_idx = 0;
    const std::int64_t bos_token_idx = 1;
    const std::int64_t eos_token_idx = 2;
    const std::int64_t pad_token_idx = 3;
    const std::array<std::string, 4> symbols = { "<unk>", "<s>", "</s>", "<pad>" };

private:
    std::map<std::string, std::int64_t> token_to_index_;
    std::vector<std::string> index_to_token_;

    void
    init_token_to_index();
};

} // fairseq2::detail
