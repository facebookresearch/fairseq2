// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>

#ifdef FAIRSEQ2N_USE_TBB
#include <oneapi/tbb.h>
#endif

namespace fairseq2n::detail {

template<typename T>
void
parallel_for(const std::function<void(T begin, T end)> &fn, T begin, T end)
{
#ifdef FAIRSEQ2N_USE_TBB
    tbb::blocked_range<T> range{begin, end};

    tbb::parallel_for(
        range, [&fn](const tbb::blocked_range<T> &r)
        {
            fn(r.begin(), r.end());
        });
#else
    // TODO: Use OpenMP!
    fn(begin, end);
#endif
}

template<typename T>
inline void
parallel_for(const std::function<void(T begin, T end)> &fn, T size)
{
    parallel_for(fn, T{}, size);
}

}  // namespace fairseq2n::detail
