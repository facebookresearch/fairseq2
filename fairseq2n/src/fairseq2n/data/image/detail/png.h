// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include <png.h>


#include "fairseq2n/memory.h"
#include "fairseq2n/span.h"
#include "fairseq2n/utils/cast.h"

namespace fairseq2n::detail {

// Wraps `memory_block` as a "virtual" file to use with libpng.
class vio_file {
public:
    explicit
    vio_file(memory_block &&block) noexcept
      : block_{std::move(block)}
    {}

    ::png_voidp get_io_ptr() {
        return static_cast<::png_voidp>(&io_ptr_);
    }

    ::png_size_t read(void *ptr, ::png_size_t size);

    ::png_size_t write(const void *ptr, ::png_size_t size);

private: 
    static std::size_t
    as_size(::png_size_t value) noexcept
    {
        return conditional_cast<std::size_t>(value);
    }

private:
    memory_block block_;
    ::png_size_t current_pos_{};
    char io_ptr_;
};

} // namespace fairseq2n::detail