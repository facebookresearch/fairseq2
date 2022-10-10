// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2 {

class FAIRSEQ2_API list_data_source final : public data_source {
public:
    explicit list_data_source(const ivalue &v) noexcept;

    bool
    move_next() override;

    ivalue
    current() const noexcept override;

    void
    reset() override;

    void
    seek(std::ptrdiff_t offset, whence w) override;

    bool
    seekable() const noexcept override;

private:
    generic_list<ivalue> list_;
    generic_list<ivalue>::iterator pos_;
    bool iterating_ = false;
};

}  // namespace fairseq2
