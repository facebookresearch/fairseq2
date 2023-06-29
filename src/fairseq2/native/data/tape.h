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

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data.h"

namespace fairseq2 {

class FAIRSEQ2_API tape {
public:
    static void
    check(bool expr)
    {
        if (!expr)
            throw_corrupt();
    }

public:
    explicit
    tape(std::vector<data> storage = {}) noexcept
      : storage_(std::move(storage))
    {}

    template <typename T>
    void
    record(const T &d);

    template <typename T>
    void
    record(const std::vector<T> &d);

    template <typename T>
    void
    record(const std::optional<T> &d);

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
        iter_ = storage_.begin();
    }

    const std::vector<data> &
    storage() const noexcept
    {
        return storage_;
    }

private:
    [[noreturn]] static void
    throw_corrupt();

private:
    std::vector<data> storage_;
    std::vector<data>::iterator iter_ = storage_.begin();
};

class FAIRSEQ2_API corrupt_tape_error : public std::logic_error {
public:
    using std::logic_error::logic_error;

public:
    corrupt_tape_error(const corrupt_tape_error &) = default;
    corrupt_tape_error &operator=(const corrupt_tape_error &) = default;

   ~corrupt_tape_error() override;
};

inline void
tape::throw_corrupt()
{
    throw corrupt_tape_error{"The tape is corrupt. The state of the data pipeline cannot be restored."};
}

}  // namespace fairseq2

#include "fairseq2/native/data/tape-inl.h"
