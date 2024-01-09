// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include <zip.h>

#include "fairseq2n/memory.h"
#include "fairseq2n/span.h"
#include "fairseq2n/data/data_source.h"

namespace fairseq2n::detail {

class zip_file_data_source final : public data_source {
public:
    explicit
    zip_file_data_source(std::string &&pathname);

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

    bool
    is_infinite() const noexcept override;

private:
    memory_block
    next_line();

    [[noreturn]] void
    handle_error();

    [[noreturn]] void
    throw_read_failure();

private:
    std::string pathname_;
    zip_t* zip_reader_;
    std::size_t num_entries_;
    std::size_t num_files_read_ = 0;
};

}  // namespace fairseq2n::detail
