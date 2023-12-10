// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

#include <ATen/Storage.h>
#include <ATen/Tensor.h>
#include <torch/version.h>

#include "fairseq2n/memory.h"

namespace fairseq2n::detail {

inline memory_span
get_raw_storage(const at::Tensor &tensor)
{
    const at::Storage &storage = tensor.storage();

    return memory_span{static_cast<const std::byte *>(storage.data()), storage.nbytes()};
}

inline writable_memory_span
get_raw_mutable_storage(const at::Tensor &tensor)
{
    const at::Storage &storage = tensor.storage();

    return writable_memory_span{
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 1)
        static_cast<std::byte *>(storage.data()), storage.nbytes()};
#else
        static_cast<std::byte *>(storage.mutable_data()), storage.nbytes()};
#endif
}

}  // namespace fairseq2n::detail
