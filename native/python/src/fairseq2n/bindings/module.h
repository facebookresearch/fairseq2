// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fairseq2n/bindings/type_casters/data.h"
#include "fairseq2n/bindings/type_casters/map_fn.h"
#include "fairseq2n/bindings/type_casters/py.h"
#include "fairseq2n/bindings/type_casters/string.h"
#include "fairseq2n/bindings/type_casters/torch.h"

namespace fairseq2n {

void
def_audio(pybind11::module_ &data_module);

void
def_image(pybind11::module_ &data_module);

void
def_data(pybind11::module_ &base_module);

void
def_data_pipeline(pybind11::module_ &data_module);

void
def_memory(pybind11::module_ &base_module);

void
def_sentencepiece(pybind11::module_ &text_module);

void
def_string(pybind11::module_ &data_module);

void
def_text(pybind11::module_ &data_module);

void
def_text_converters(pybind11::module_ &text_module);

void
def_text_reader(pybind11::module_ &text_module);

}  // namespace fairseq2n
