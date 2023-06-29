// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/detail/file_system.h"

#include <array>
#include <memory>
#include <system_error>

#include <fnmatch.h>
#include <fts.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fmt/core.h>
#include <strnatcmp.h>

#include "fairseq2/native/error.h"

namespace fairseq2::detail {
namespace {

struct fts_deleter {
    void
    operator()(::FTS *fts)
    {
        if (fts != nullptr)
            ::fts_close(fts);
    }
};

inline int
natural_sort(const ::FTSENT **a, const ::FTSENT **b)
{
    return ::strnatcmp((*a)->fts_name, (*b)->fts_name);
}

auto
make_fts(const std::string &pathname)
{
    std::array<const char *, 2> a{pathname.c_str(), nullptr};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto *p = const_cast<char * const *>(a.data());

    ::FTS *fts = ::fts_open(p, FTS_LOGICAL | FTS_NOCHDIR, natural_sort);
    if (fts == nullptr)
        throw std::system_error{last_error(),
            fmt::format("'{}' cannot be traversed", pathname)};

    return std::unique_ptr<::FTS, fts_deleter>(fts);
}

}  // namespace

std::vector<data>
list_files(const std::string &pathname, const std::optional<std::string> &pattern)
{
    auto fts = make_fts(pathname);

    std::vector<data> output{};

    ::FTSENT *e = nullptr;
    while ((e = ::fts_read(fts.get())) != nullptr) {
        if (e->fts_info == FTS_ERR || e->fts_info == FTS_DNR || e->fts_info == FTS_NS)
            throw std::system_error{last_error(),
                fmt::format("'{}' cannot be traversed", e->fts_accpath)};

        if (e->fts_info != FTS_F)
            continue;

        // We only return regular and block files.
        if (!S_ISREG(e->fts_statp->st_mode) && !S_ISBLK(e->fts_statp->st_mode))
            continue;

        if (pattern && !pattern->empty()) {
            int r = ::fnmatch(pattern->c_str(), e->fts_accpath, 0);
            if (r == FNM_NOMATCH)
                continue;
            if (r != 0)
                throw std::invalid_argument{"pattern is invalid."};
        }

        output.emplace_back(e->fts_accpath);
    }

    std::error_code err = last_error();
    if (err)
        throw std::system_error{err,
            fmt::format("'{}' cannot be traversed", pathname)};

    return output;
}

}  // namespace fairseq2::detail
