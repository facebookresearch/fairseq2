// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include <fairseq2/native/data/data_processor.h>

namespace fairseq2 {

std::shared_ptr<data_processor>
as_data_processor(pybind11::handle h);

}  // namespace fairseq2
