// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/image/detail/png_read_struct.h"

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

png_read::png_read() {
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png_ptr == nullptr) {
        throw internal_error("Failed to create PNG read struct.");
    }
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == nullptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        throw internal_error("Failed to create PNG info struct.");
    }
}

png_read::~png_read() {
    if (png_ptr != nullptr) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    }
}

png_structp png_read::getPngPtr() const {
    return png_ptr;
}

png_infop png_read::getInfoPtr() const {
    return info_ptr;
}

} // namespace fairseq2n::detail
