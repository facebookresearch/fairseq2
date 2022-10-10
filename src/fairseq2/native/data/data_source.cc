// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/data_source.h"

#include "fairseq2/native/exception.h"
#include "fairseq2/native/data/fs.h"
#include "fairseq2/native/data/list_data_source.h"

namespace fairseq2 {

data_source::~data_source() = default;

intrusive_ptr<data_source>
data_source::list_files(array_view<std::string> paths, const std::optional<std::string> &pattern)
{
    generic_list<std::string> files = detail::list_files(paths, pattern);

    // mark used

    return make_intrusive<list_data_source>(files);
}

void
data_source::seek(std::ptrdiff_t, whence)
{
    throw not_supported_error{"Seek operation is not supported."};
}

bool
data_source::seekable() const noexcept
{
    return false;
}

}  // namespace fairseq2
