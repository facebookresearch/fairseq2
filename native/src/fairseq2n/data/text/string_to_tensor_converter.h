// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

namespace fairseq2n {

class immutable_string;

class FAIRSEQ2_API string_to_tensor_converter final {
public:
    explicit
    string_to_tensor_converter(
        std::vector<std::int64_t> size = {}, std::optional<at::ScalarType> maybe_dtype = {});

    data
    operator()(data &&d) const;

private:
    template <typename T>
    void
    fill_storage(at::Tensor &tensor, const std::vector<immutable_string> &strings) const;

private:
    std::vector<std::int64_t> size_;
    at::ScalarType dtype_;
};

}  // namespace fairseq2n
