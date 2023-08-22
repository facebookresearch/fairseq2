// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdexcept>

#include "fairseq2n/api.h"

namespace fairseq2n {

class FAIRSEQ2_API internal_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

public:
    internal_error(const internal_error &) = default;
    internal_error &operator=(const internal_error &) = default;

   ~internal_error() override;
};

class FAIRSEQ2_API not_supported_error : public std::domain_error {
public:
    using std::domain_error::domain_error;

public:
    not_supported_error(const not_supported_error &) = default;
    not_supported_error &operator=(const not_supported_error &) = default;

   ~not_supported_error() override;
};

}  // namespace fairseq2n
