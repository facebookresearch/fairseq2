// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <csignal>
#include <system_error>
#include <thread>
#include <utility>

#include <pthread.h>

namespace fairseq2n::detail {

template<typename Func, typename... Args>
std::thread
start_thread(Func &&f, Args &&...args)
{
    ::sigset_t mask{};
    ::sigset_t original_mask{};

    sigfillset(&mask);

    // Block all async signals in the new thread.
    int result = ::pthread_sigmask(SIG_SETMASK, &mask, &original_mask);
    if (result != 0)
        throw std::system_error{result, std::generic_category()};

    std::thread t{std::forward<Func>(f), std::forward<Args>(args)...};

    // Restore the signal mask.
    result = ::pthread_sigmask(SIG_SETMASK, &original_mask, nullptr);
    if (result != 0)
        throw std::system_error{result, std::generic_category()};

    return t;
}

}  // namespace fairseq2n::detail
