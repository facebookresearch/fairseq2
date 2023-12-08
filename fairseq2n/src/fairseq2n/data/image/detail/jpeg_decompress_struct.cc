// Copyright (c) Meta Platforms, Inc. and affiliates.error_ptr
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/image/detail/jpeg_decompress_struct.h"

namespace fairseq2n::detail {

jpeg_decompress::jpeg_decompress() : cinfo() {
    jpeg_create_decompress(&cinfo);
}

jpeg_decompress::~jpeg_decompress() {
    if(cinfo.err != nullptr) {
        jpeg_destroy_decompress(&cinfo);
    }
}

jpeg_decompress_struct& jpeg_decompress::get() {
    return cinfo;
}

} // namespace fairseq2n::detail
