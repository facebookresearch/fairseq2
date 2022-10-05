// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data_source.h"

namespace fairseq2 {

class FAIRSEQ2_API list_data_source : public data_source {
public:
    explicit list_data_source(const c10::IValue &v) noexcept;

    bool
    move_next() override;

    const c10::IValue &
    current() const noexcept override;

    void
    reset() override;

private:
    c10::List<c10::IValue> list_;
    c10::List<c10::IValue>::iterator pos_;
};

}  // namespace fairseq2
