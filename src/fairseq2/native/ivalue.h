// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>

namespace fairseq2 {

using ivalue = at::IValue;

template <class T>
using generic_list = at::List<T>;

template <class K, class V>
using generic_dict = at::Dict<K, V>;

}  // namespace fairseq2
