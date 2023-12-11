// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdexcept>

#include "fairseq2n/api.h"
#include "fairseq2n/memory.h"

namespace fairseq2n {

class FAIRSEQ2_API byte_stream {
public:
    byte_stream() noexcept = default;

    byte_stream(const byte_stream &) = default;
    byte_stream &operator=(const byte_stream &) = default;

    byte_stream(byte_stream &&) = default;
    byte_stream &operator=(byte_stream &&) = default;

    virtual
   ~byte_stream();

    virtual memory_block
    read_chunk() = 0;

    virtual void
    reset() = 0;
};

class FAIRSEQ2_API byte_stream_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

public:
    byte_stream_error(const byte_stream_error &) = default;
    byte_stream_error &operator=(const byte_stream_error &) = default;

   ~byte_stream_error() override;
};

}  // namespace fairseq2n
