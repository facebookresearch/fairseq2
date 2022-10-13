// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "fairseq2/native/api.h"
#include "fairseq2/native/array_view.h"
#include "fairseq2/native/intrusive_ptr.h"
#include "fairseq2/native/ivalue.h"

namespace fairseq2 {

enum class whence { begin, current, end };

class FAIRSEQ2_API data_source : public intrusive_ptr_target {
public:
    static intrusive_ptr<data_source>
    list_files(array_view<std::string> paths, const std::optional<std::string> &pattern);

    virtual ~data_source() override;

    virtual bool
    move_next() = 0;

    virtual ivalue
    current() const noexcept = 0;

    virtual void
    reset() = 0;

    virtual void
    seek(std::ptrdiff_t offset, whence w = whence::begin);

    virtual bool
    seekable() const noexcept;
};

}  // namespace fairseq2
