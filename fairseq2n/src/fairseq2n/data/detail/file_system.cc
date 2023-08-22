// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/detail/file_system.h"

#include <array>
#include <memory>
#include <system_error>

#include <fnmatch.h>
#include <fts.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <strnatcmp.h>

#include "fairseq2n/detail/error.h"
#include "fairseq2n/detail/exception.h"

namespace fairseq2n::detail {
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
    std::array<const char *, 2> arr{pathname.c_str(), nullptr};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto *ptr = const_cast<char * const *>(arr.data());

    ::FTS *fts = ::fts_open(ptr, FTS_LOGICAL | FTS_NOCHDIR, natural_sort);
    if (fts == nullptr)
        throw_system_error(last_error(),
            "'{}' cannot be traversed", pathname);

    return std::unique_ptr<::FTS, fts_deleter>(fts);
}

}  // namespace

data_list
list_files(const std::string &pathname, const std::optional<std::string> &maybe_pattern)
{
    auto fts = make_fts(pathname);

    data_list output{};

    while (::FTSENT *ent = ::fts_read(fts.get())) {
        if (ent->fts_info == FTS_ERR || ent->fts_info == FTS_DNR || ent->fts_info == FTS_NS)
            throw_system_error(last_error(),
                "'{}' cannot be traversed", ent->fts_accpath);

        if (ent->fts_info != FTS_F)
            continue;

        // We only return regular and block files.
        if (!S_ISREG(ent->fts_statp->st_mode) && !S_ISBLK(ent->fts_statp->st_mode))
            continue;

        if (maybe_pattern) {
            int result = ::fnmatch(maybe_pattern->c_str(), ent->fts_accpath, 0);

            if (result == FNM_NOMATCH)
                continue;

            if (result != 0)
                throw_<std::invalid_argument>("`pattern` is invalid.");
        }

        output.emplace_back(ent->fts_accpath);
    }

    std::error_code err = last_error();
    if (err)
        throw_system_error(err,
            "'{}' cannot be traversed", pathname);

    return output;
}

}  // namespace fairseq2n::detail
