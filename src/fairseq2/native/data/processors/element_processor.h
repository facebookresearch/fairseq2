// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <string_view>
#include <utility>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_processor.h"
#include "fairseq2/native/data/element_selector.h"

namespace fairseq2 {

class FAIRSEQ2_API element_processor final : public data_processor {
public:
    explicit
    element_processor(std::shared_ptr<const data_processor> p, std::string_view selector)
      : processor_{std::move(p)}, selector_{selector}
    {}

    data
    process(data &&d) const override;

private:
    std::shared_ptr<const data_processor> processor_;
    detail::element_selector selector_;
};

}  // namespace fairseq2
