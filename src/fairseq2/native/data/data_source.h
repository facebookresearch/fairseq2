// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>

#include <ATen/ATen.h>

#include "fairseq2/native/api.h"

namespace fairseq2 {

class FAIRSEQ2_API data_source : public c10::intrusive_ptr_target {
public:
    virtual ~data_source() override;

    virtual bool
    move_next() = 0;

    virtual const c10::IValue &
    current() const noexcept = 0;

    virtual void
    reset() = 0;

    static c10::intrusive_ptr<data_source>
    list_files(c10::ArrayRef<std::string> paths, const std::optional<std::string> &pattern);
};

}  // namespace fairseq2
