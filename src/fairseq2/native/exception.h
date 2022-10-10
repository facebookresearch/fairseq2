// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdexcept>

#include "fairseq2/native/api.h"

namespace fairseq2 {

class FAIRSEQ2_API not_supported_error : public std::logic_error {
public:
    using std::logic_error::logic_error;

    ~not_supported_error() override;
};

}  // namespace fairseq2
