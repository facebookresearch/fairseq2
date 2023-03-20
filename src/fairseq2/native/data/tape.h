// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "fairseq2/native/api.h"
#include "fairseq2/native/data/data.h"
#include "fairseq2/native/utils/cast.h"

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

    void
    record(const data &d);

    void
    record_if(const std::optional<data> &d)
    {
        if (d) {
            record(true);

            record(*d);
        } else
            record(false);
    }

    data
    read();

    template <typename T>
    T
    read();

    std::optional<data>
    read_if()
    {
        if (read<bool>())
            return read();

        return {};
    }

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

template <typename T>
T
tape::read()
{
    data d = read();

    if constexpr (std::is_same_v<T, bool>) {
        if (d.is_bool())
            return d.as_bool();

        throw_corrupt();
    }

    if constexpr (std::is_integral_v<T>) {
        if (T i{}; d.is_int() && detail::try_narrow(d.as_int(), i))
            return i;

        throw_corrupt();
    }
}

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
    throw corrupt_tape_error{"The tape is corrupt."};
}

}  // namespace fairseq2
