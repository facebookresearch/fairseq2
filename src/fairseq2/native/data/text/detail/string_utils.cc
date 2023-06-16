// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/text/detail/string_utils.h"

#include <ATen/Functions.h>
#include <stdexcept>
#include <string>

namespace fairseq2::detail {

at::Tensor
parse_tensor(const immutable_string &input)
{
    auto tokens = input.split(' ');
    auto tensor = at::empty(static_cast<std::int64_t>(tokens.size()), at::TensorOptions().dtype(at::kLong));

    auto tensor_a = tensor.accessor<std::int64_t, 1>();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        try {
            tensor_a[static_cast<std::int64_t>(i)] = tokens[i].to_int32();
        }
        catch(const std::runtime_error &) {
            throw std::invalid_argument("Non integer token encountered in input string: " + tokens[i].to_string());
        } catch (const std::exception &e) {
            throw e;
        }
    }

    return tensor;
}

} // fairseq2::detail
