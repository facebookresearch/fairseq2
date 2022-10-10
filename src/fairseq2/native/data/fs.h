// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <optional>
#include <string>

#include "fairseq2/native/array_view.h"
#include "fairseq2/native/ivalue.h"

namespace fairseq2::detail {

generic_list<std::string>
list_files(array_view<std::string> paths, const std::optional<std::string> &pattern);

}  // namespace fairseq2::detail
