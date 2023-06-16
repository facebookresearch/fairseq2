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

#include "fairseq2/native/memory.h"
#include "fairseq2/native/span.h"
#include "fairseq2/native/data/data_source.h"
#include <zip.h>

namespace fairseq2 {
namespace detail {

class zipfile_data_source final : public data_source {
public:
    explicit
    zipfile_data_source(std::string &&pathname);

    std::optional<data>
    next() override;

    void
    reset() override;

    void
    record_position(tape &t) const override;

    void
    reload_position(tape &t) override;

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

}  // namespace fairseq2::detail

class data_pipeline_builder;

FAIRSEQ2_API data_pipeline_builder
read_zipped_records(std::string pathname);

}  // namespace fairseq2
