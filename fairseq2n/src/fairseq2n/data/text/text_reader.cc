// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/text_reader.h"

#include <memory>

#include "fairseq2n/data/data_pipeline.h"
#include "fairseq2n/data/text/text_data_source.h"

using namespace fairseq2n::detail;

namespace fairseq2n {

data_pipeline_builder
read_text(std::string pathname, text_options opts)
{
    auto factory = [pathname = std::move(pathname), opts = std::move(opts)]() mutable
    {
        return std::make_unique<text_data_source>(std::move(pathname), std::move(opts));
    };

    return data_pipeline_builder{std::move(factory)};
}

}  // namespace fairseq2n
