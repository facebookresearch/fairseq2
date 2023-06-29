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
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class FAIRSEQ2_API string_splitter final : public data_processor {
public:
    explicit
    string_splitter(char sep = '\t', std::optional<std::vector<std::string>> names = {}) noexcept;

    data
    process(data &&d) const override;

private:
    char sep_;
    std::vector<std::string> names_;
};

}  // namespace fairseq2
