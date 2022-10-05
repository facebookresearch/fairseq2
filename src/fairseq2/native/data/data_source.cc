// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/data_source.h"

#include "fairseq2/native/data/fs.h"
#include "fairseq2/native/data/list_data_source.h"

namespace fairseq2 {

data_source::~data_source() = default;

c10::intrusive_ptr<data_source>
data_source::list_files(c10::ArrayRef<std::string> paths, const std::optional<std::string> &pattern)
{
    c10::List<std::string> files = detail::list_files(paths, pattern);

    // mark used

    return c10::make_intrusive<list_data_source>(files);
}

}  // namespace fairseq2
