// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <vector>

#include "fairseq2/native/data/data.h"

namespace fairseq2::detail {

std::vector<data>
list_files(const std::string &pathname, const std::string &pattern);

}
