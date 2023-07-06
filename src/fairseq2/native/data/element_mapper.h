// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string_view>
#include <utility>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_pipeline.h"
#include "fairseq2/native/data/element_selector.h"

namespace fairseq2 {

class FAIRSEQ2_API element_mapper {
public:
    explicit
    element_mapper(map_fn fn, std::optional<std::string_view> selector = {})
      : map_fn_{std::move(fn)}
    {
        if (selector)
            selector_ = detail::element_selector{*selector};
    }

    data
    operator()(data &&d);

private:
    map_fn map_fn_;
    std::optional<detail::element_selector> selector_;
};

}  // namespace fairseq2
