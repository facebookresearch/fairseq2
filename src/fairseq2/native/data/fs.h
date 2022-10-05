// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <optional>
#include <string>

#include <ATen/ATen.h>

namespace fairseq2::detail {

c10::List<std::string>
list_files(c10::ArrayRef<std::string> paths, const std::optional<std::string> &pattern);

}  // namespace fairseq2::detail
