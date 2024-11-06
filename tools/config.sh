#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =================================================================================
# Configurable variables
# =================================================================================

# === Installation === #

# change this to true to skip apex installation
skip_apex=false

# change this to the name of the environment you want to create
default_env_name=fairseq2


# === Packages === #

# change this to the version of cuda you want to install
cuda_version=12.1

# change this to the version of torch you want to install
torch_version=2.4.0

# change this to the version of python you want to install
python_version=3.10.14

# change this to the version of libsndfile you want to install
libsndfile_version=1.0.31

# change this to the version of tensorboard you want to install
tensorboard_version=2.18.0


# === Paths === #

# change this to where you want to store the environment
# e.g. $HOME/fairseq2
env_base_path=
