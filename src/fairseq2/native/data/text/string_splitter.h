// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2 {

class FAIRSEQ2_API string_splitter final {
public:
    explicit
    string_splitter(
        char separator = '\t', std::optional<std::vector<std::string>> names = {}) noexcept;

    data
    operator()(data &&d) const;

private:
    char separator_;
    std::vector<std::string> names_;
};

}  // namespace fairseq2
