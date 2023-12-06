// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdio>
// Forward declaration
//using FILE = struct _IO_FILE;
#include <jpeglib.h>

#include "fairseq2n/exception.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {

class jpeg_decompress {
public:
    jpeg_decompress();
    ~jpeg_decompress();
    jpeg_decompress_struct& get();
    jpeg_decompress(const jpeg_decompress&) = delete; 
    jpeg_decompress& operator=(const jpeg_decompress&) = delete; 

private:
    jpeg_decompress_struct cinfo;
};

} // namespace fairseq2n::detail
