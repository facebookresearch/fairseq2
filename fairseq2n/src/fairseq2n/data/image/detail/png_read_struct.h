// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <png.h>

namespace fairseq2n::detail {

class png_read{
public:
    png_read();
    ~png_read();

    png_structp getPngPtr() const;
    png_infop getInfoPtr() const;

private:
    png_structp png_ptr;
    png_infop info_ptr;
    png_read(const png_read&);
    png_read& operator=(const png_read&); 
};

} // namespace fairseq2n::detail
