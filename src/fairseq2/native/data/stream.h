// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdexcept>

#include "fairseq2/native/api.h"
#include "fairseq2/native/memory.h"

namespace fairseq2 {

class FAIRSEQ2_API stream {
public:
    stream() noexcept = default;

    stream(const stream &) = default;
    stream &operator=(const stream &) = default;

    stream(stream &&) = default;
    stream &operator=(stream &&) = default;

    virtual
   ~stream();

    virtual memory_block
    read_chunk() = 0;

    virtual void
    reset() = 0;
};

class FAIRSEQ2_API stream_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

public:
    stream_error(const stream_error &) = default;
    stream_error &operator=(const stream_error &) = default;

   ~stream_error() override;
};

}  // namespace fairseq2
