// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/data/fs.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

#include <fnmatch.h>
#include <fts.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fmt/format.h>
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

std::vector<const char *>
get_c_strs(array_view<std::string> paths)
{
    std::vector<const char *> c_strs{};

    std::transform(paths.begin(), paths.end(), std::back_inserter(c_strs),
                   [](const std::string &p) {
                       return p.c_str();
                   });

    c_strs.push_back(nullptr);

    return c_strs;
}

inline int
natural_sort(const ::FTSENT **a, const ::FTSENT **b)
{
    return ::strnatcmp((*a)->fts_name, (*b)->fts_name);
}

auto
make_fts(array_view<std::string> paths)
{
    std::vector<const char *> c_strs = get_c_strs(paths);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const auto *path_argv = const_cast<char *const *>(c_strs.data());

    ::FTS *fts = ::fts_open(path_argv, FTS_LOGICAL | FTS_NOCHDIR, natural_sort);
    if (fts == nullptr)
        throw std::system_error{last_error(), "The specified path(s) cannot be traversed."};

    return std::unique_ptr<::FTS, fts_deleter>(fts);
}

}  // namespace

generic_list<std::string>
list_files(array_view<std::string> paths, const std::optional<std::string> &pattern)
{
    auto fts = make_fts(paths);

    generic_list<std::string> result{};

    ::FTSENT *e{};
    while ((e = ::fts_read(fts.get())) != nullptr) {
        if (e->fts_info == FTS_ERR || e->fts_info == FTS_DNR || e->fts_info == FTS_NS) {
            auto msg = fmt::format("The file or directory '{0}' cannot be opened.", e->fts_accpath);
            throw std::system_error{last_error(), msg};
        }

        if (e->fts_info != FTS_F)
            continue;

        // We only return regular and block files.
        if (!S_ISREG(e->fts_statp->st_mode) && !S_ISBLK(e->fts_statp->st_mode))
            continue;

        if (pattern != std::nullopt && !pattern->empty()) {
            int r = ::fnmatch(pattern->c_str(), e->fts_accpath, 0);
            if (r == FNM_NOMATCH)
                continue;
            if (r != 0)
                throw std::invalid_argument{"The pattern cannot be used for comparison."};
        }

        result.emplace_back(e->fts_accpath);
    }

    std::error_code err = last_error();
    if (err)
        throw std::system_error{err, "The specified path(s) cannot be traversed."};

    return result;
}

}  // namespace fairseq2::detail
