// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include <pybind11/pybind11.h>

namespace fairseq2 {

class data_processor;

namespace detail {

std::shared_ptr<const data_processor>
as_data_processor(const pybind11::object &fn, std::optional<std::string_view> selector);

}  // namespace detail
}  // namespace fairseq2
