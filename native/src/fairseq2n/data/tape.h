// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fairseq2n/api.h"
#include "fairseq2n/data/data.h"

namespace fairseq2n {

class FAIRSEQ2_API tape {
public:
    explicit
    tape(data_list storage = {}) noexcept
      : storage_(std::move(storage))
    {}

    template <typename T>
    void
    record(const T &value);

    template <typename T>
    void
    record(const std::vector<T> &value);

    template <typename T>
    void
    record(const std::optional<T> &maybe_value);

    void
    record_data(const data &d);

    template <typename T>
    T
    read();

    data
    read_data();

    void
    rewind() noexcept
    {
        pos_ = storage_.begin();
    }

    bool
    is_eod() const noexcept
    {
        return pos_ == storage_.end();
    }

    const data_list &
    storage() const noexcept
    {
        return storage_;
    }

private:
    [[noreturn]] static void
    throw_corrupt();

private:
    data_list storage_;
    data_list::iterator pos_ = storage_.begin();
};

inline void
tape::throw_corrupt()
{
    throw std::invalid_argument(
        "The tape is corrupt. The state of the data pipeline cannot be restored.");
}

}  // namespace fairseq2n

#include "fairseq2n/data/tape-inl.h"
