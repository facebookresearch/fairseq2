// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class FAIRSEQ2_API list_processor final : public data_processor {
public:
    explicit
    list_processor(
        std::vector<std::shared_ptr<const data_processor>> processors,
        std::optional<std::vector<std::size_t>> indices = {},
        bool disable_parallelism = false);

    data
    operator()(const data &d) const override;

    data
    operator()(data &&d) const override;

private:
    void
    validate(const data &d) const;

    template <typename F>
    void
    parallel_for(F &f, std::size_t n) const;

private:
    std::vector<std::shared_ptr<const data_processor>> processors_;
    std::vector<std::size_t> indices_{};
    bool disable_parallelism_;
};

}  // namespace fairseq2
