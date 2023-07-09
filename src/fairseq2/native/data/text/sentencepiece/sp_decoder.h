// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <vector>

#include <ATen/Tensor.h>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2 {
namespace detail {

class decoder_op;

}

class sp_model;

class FAIRSEQ2_API sp_decoder final {
    friend class detail::decoder_op;

public:
    explicit
    sp_decoder(std::shared_ptr<const sp_model> model, bool reverse = false) noexcept;

    data
    operator()(data &&d) const;

private:
    data_list
    decode(at::Tensor &&tensor) const;

private:
    std::shared_ptr<const sp_model> model_;
    bool reverse_;
};

}  // namespace fairseq2
