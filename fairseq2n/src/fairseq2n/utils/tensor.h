// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <vector>

#include <ATen/Tensor.h>

namespace fairseq2n::detail {

template <typename T>
inline at::Tensor
make_tensor_from_vector(
    const std::vector<T> &src,
    const std::initializer_list<std::int64_t> &shape) noexcept
{
    auto storage = std::make_shared<std::vector<T>>(src);

    return at::from_blob(
        storage->data(),
        c10::ArrayRef<std::int64_t>(shape),
        [storage](void*) mutable { storage.reset(); }
    );
}

}  // namespace fairseq2::detail
