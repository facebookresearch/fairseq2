// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/text_splitter.h"

namespace fairseq2 {

data
text_splitter::operator()(data &&d) const
{
    if (!d.is_string())
        throw std::invalid_argument{"The input data must be of type string."};

    std::vector<data> parts{};

    d.as_string().split(separator_, [&parts](immutable_string &&s) {
        parts.emplace_back(std::move(s));
    });

    return parts;
}

}  // namespace fairseq2
