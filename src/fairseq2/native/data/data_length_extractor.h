// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/element_selector.h"

namespace fairseq2 {

class data;

class FAIRSEQ2_API data_length_extractor {
public:
    explicit
    data_length_extractor(std::optional<std::string_view> selector)
    {
        if (selector)
            selector_ = detail::element_selector{*selector};
    }

    std::size_t
    operator()(const data &d) const;

private:
    std::optional<detail::element_selector> selector_{};
};

}  // namespace fairseq2
