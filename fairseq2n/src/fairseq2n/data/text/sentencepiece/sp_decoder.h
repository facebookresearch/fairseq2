// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <vector>

#include <ATen/Tensor.h>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"
#include "fairseq2n/data/immutable_string.h"

namespace fairseq2n {

class sp_model;

class FAIRSEQ2_API sp_decoder final {
public:
    explicit
    sp_decoder(std::shared_ptr<const sp_model> model, bool reverse = false) noexcept;

    data
    operator()(data &&d) const;

    data
    decode_from_tokens(data &&d) const;

private:
    template <typename T>
    immutable_string
    decode(const at::Tensor &tensor) const;

private:
    std::shared_ptr<const sp_model> model_;
    bool reverse_;
};

}  // namespace fairseq2n
