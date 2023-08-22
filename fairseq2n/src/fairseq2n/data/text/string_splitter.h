// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

namespace fairseq2n {

class FAIRSEQ2_API string_splitter final {
public:
    explicit
    string_splitter(
        char separator = '\t',
        std::vector<std::string> names = {},
        std::vector<std::size_t> indices = {},
        bool exclude = false);

    data
    operator()(data &&d) const;

private:
    char separator_;
    std::vector<std::string> names_;
    std::vector<std::size_t> indices_;
    bool exclude_;
};

}  // namespace fairseq2n
