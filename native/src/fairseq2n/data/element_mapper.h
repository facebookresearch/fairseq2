// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>
#include <utility>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/element_selector.h"

namespace fairseq2n {

class FAIRSEQ2_API element_mapper {
public:
    explicit
    element_mapper(map_fn fn, std::optional<std::string> maybe_selector = {});

    data
    operator()(data &&d);

private:
    map_fn map_fn_;
    std::optional<element_selector> maybe_selector_{};
};

}  // namespace fairseq2n
