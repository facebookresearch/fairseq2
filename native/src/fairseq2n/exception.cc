// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/exception.h"

namespace fairseq2n {

internal_error::~internal_error() = default;

not_supported_error::~not_supported_error() = default;

}
