// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/image/detail/png.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

::png_size_t vio_file::read(void *ptr, ::png_size_t size) {
    if (current_pos_ >= block_.size()) {
        return 0;
    }

    ::png_size_t bytes_to_read = std::min(size, block_.size() - current_pos_);
    std::memcpy(ptr, block_.data() + current_pos_, bytes_to_read);
    current_pos_ += bytes_to_read;

    return bytes_to_read;
}

::png_size_t vio_file::write(const void *ptr, ::png_size_t size) {
     // We only support decoding image files.
    return -1;
}

}

  // namespace fairseq2n::detail
