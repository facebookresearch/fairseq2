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

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_processor.h"

namespace fairseq2 {

class immutable_string;

class FAIRSEQ2_API string_to_tensor_converter final : public data_processor {
public:
    explicit
    string_to_tensor_converter(
        std::optional<std::vector<std::int64_t>> size = {},
        std::optional<at::ScalarType> dtype = {});

    data
    process(data &&d) const override;

private:
    template <typename T>
    void
    fill_storage(at::Tensor &t, const std::vector<immutable_string> &strs) const;

private:
    std::optional<std::vector<std::int64_t>> size_;
    at::ScalarType dtype_;
};

}  // namespace fairseq2
