// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string_view>
#include <utility>

#include <unistd.h>

#include "fairseq2n/memory.h"

namespace fairseq2n::detail {

constexpr int invalid_fd = -1;

class file_desc {
public:
    file_desc() noexcept = default;

    file_desc(int fd) noexcept
      : fd_{fd}
    {}

    file_desc(const file_desc &) = delete;
    file_desc &operator=(const file_desc &) = delete;

    file_desc(file_desc &&other) noexcept
      : fd_{other.fd_}
    {
        other.fd_ = invalid_fd;
    }

    file_desc &
    operator=(file_desc &&other) noexcept
    {
        close_fd();

        fd_ = std::exchange(other.fd_, invalid_fd);

        return *this;
    }

   ~file_desc()
    {
        close_fd();
    }

    int
    get() const noexcept
    {
        return fd_;
    }

private:
    void
    close_fd() noexcept
    {
        if (fd_ == invalid_fd)
            return;

        ::close(fd_);

        fd_ = invalid_fd;
    }

private:
    int fd_ = invalid_fd;
};

inline bool
operator==(const file_desc &lhs, const file_desc &rhs) noexcept
{
    return lhs.get() == rhs.get();
}

inline bool
operator!=(const file_desc &lhs, const file_desc &rhs) noexcept
{
    return lhs.get() != rhs.get();
}

memory_block
memory_map_file(const file_desc &fd, std::string_view pathname);

}
