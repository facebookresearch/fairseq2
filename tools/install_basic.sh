#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# =============================================================================
# Last updated: 2024-11-06
# -----------------------------------------------------------------------------
# This scripts installs fairseq2 and fairseq2-ext, and optionally APEX.
# -----------------------------------------------------------------------------
# Packages to be installed:
# - fairseq2
# - fairseq2-ext
# - APEX (optional)
# -----------------------------------------------------------------------------
# Dependencies versions to be defined:
# - PyTorch: 2.4.0 by default
# - CUDA: 12.1 by default
# -----------------------------------------------------------------------------
# Artifacts:
# - conda environment in $env_path
#   - fairseq2: editable install from source
#   - fairseq2-ext: editable install from source
#   - APEX: install from source (if not skipped)
# - symlink:
#   - $default_conda_dir/envs/$env_name -> $env_path/conda
# =============================================================================

set -eo pipefail


# =================================================================================
# Configurable variables
# =================================================================================

# change this to the name of the environment you want to create
default_env_name=fairseq2
# change this to true to skip apex installation
skip_apex=false
# change this to the version of cuda you want to install
cuda_version=12.1
# change this to the version of torch you want to install
torch_version=2.4.0
# change this to the version of python you want to install
python_version=3.10.14
# change this to the version of libsndfile you want to install
libsndfile_version=1.0.31
# change this to where you want to store the environment
env_base_path=$HOME/fairseq2/envs


# =================================================================================
# Utility functions
# =================================================================================

usage() {
    echo "Usage: $0 ENV_NAME [--skip-apex]"
}

get_env_path() {
    local env_name=$1
    local env_base_path=$2
    local default_env_name=$3

    if [[ -z "$env_name" ]]; then
        env_name=$default_env_name
    fi
    
    if [[ -z "$env_name" ]]; then
        # in case env_name is still empty
        # explicitly log the error
        usage
        echo "Or set default_env_name in config.sh"
        return 1
    fi

    # get the env_path
    if [[ -z "$env_base_path" ]]; then
        echo "env_base_path is not set, please set env_base_path manually"
        return 1
    fi
    env_path=$env_base_path/$env_name
    echo $env_path
}


check_prerequisites() {
    local env_path=$1
    local symlink_path=$2
    
    if [[ -d "$env_path" ]]; then
        echo "Directory \"$env_path\" already exists!" >&2
        return 1
    fi

    if [[ -d "$symlink_path" ]]; then
        echo "Symlink \"$symlink_path\" already exists!" >&2
        return 1
    fi

    if ! command -v conda &> /dev/null; then
        echo "Conda not found! Please install Conda and make sure that it is available in your PATH." >&2
        return 1
    fi
}

# install fairseq2
install_fairseq2() {
    local env_path=$1
    local torch_version=$2
    local cuda_version=$3

    echo "Installing fairseq2n nightly..."

    conda run --prefix "$env_path/conda" --no-capture-output --live-stream\
        pip install fairseq2n --pre --upgrade\
            --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt$torch_version/cu${cuda_version//.}


    echo "Installing fairseq2..."

    git clone git@github.com:facebookresearch/fairseq2.git

    cd fairseq2

    conda run --prefix "$env_path/conda" --no-capture-output --live-stream\
        pip install --editable .

    cd -
}

# install fairseq2-ext
install_fairseq2_ext() {
    local env_path=$1
    echo "Installing fairseq2-ext (internal extensions)..."

    git clone git@github.com:fairinternal/fairseq2-ext.git

    cd fairseq2-ext

    conda run --prefix "$env_path/conda" --no-capture-output --live-stream\
        pip install --editable .

    cd -
}


# install apex
install_apex() {
    local env_path=$1
    local cluster=$2
    local cuda_version=$3

    echo "Installing APEX... This can take a while!"

    # check if cuda is available
    if ! command -v nvcc &> /dev/null; then
        echo "CUDA not found! Please install CUDA and make sure that it is available in your PATH." >&2
        return 1
    fi

    # check nvcc version
    nvcc_version=$(command -v nvcc | xargs nvcc --version | grep -oP 'V\d+\.\d+')
    if [[ "$nvcc_version" != "V$cuda_version" ]]; then
        echo "nvcc version is not V$cuda_version! Please install CUDA $cuda_version and make sure that it is available in your PATH." >&2
        return 1
    fi

    git clone git@github.com:NVIDIA/apex.git

    cd apex

    git checkout 23.08

    conda run --prefix "$env_path/conda" --no-capture-output --live-stream\
        pip install\
            --verbose\
            --disable-pip-version-check\
            --no-cache-dir\
            --no-build-isolation\
            --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext"\
            .

    cd -
}

install() {
    # if env_name is not provided, use the default_env_name in config.sh
    local env_name=$1

    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--skip-apex)
                skip_apex=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    # get the env_path
    env_path=$(get_env_path $env_name $env_base_path $default_env_name)

    # get the conda environment directory (for symlink)
    envs_dirs=$(conda info --json | jq ".envs_dirs[0]" | xargs echo)
    symlink_path="$envs_dirs/$env_name"
    if ! check_prerequisites "$env_path" "$symlink_path"; then
        exit 1
    fi

    mkdir --parents "$env_path"

    echo "Creating '$env_name' Conda environment with PyTorch $torch_version and CUDA $cuda_version..."

    conda create\
        --prefix "$env_path/conda"\
        --yes\
        --strict-channel-priority\
        --override-channels\
        --channel conda-forge\
        python==$python_version\
        libsndfile==$libsndfile_version
    
    # check if the environment is created, if not successful, clean up and exit
    if [[ ! $? -eq 0 ]]; then
        cleanup $env_name $env_path
        exit 1
    fi

    # sym link prefix based environment to access it using the alias name
    ln -s "$env_path/conda" "$symlink_path"
    echo "Symlink $env_path/conda -> $symlink_path was created"

    # install pytorch
    conda run --prefix "$env_path/conda" --no-capture-output --live-stream\
        pip install torch==$torch_version torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    cd "$env_path"

    # installation
    install_fairseq2 "$env_path" "$torch_version" "$cuda_version"
    install_fairseq2_ext "$env_path"
    if [[ "$skip_apex" == "false" ]]; then
        install_apex "$env_path" "$cluster" "$cuda_version"
    fi

    echo -e "\n\nDone!"
    echo -e "To activate the environment, run 'conda activate $env_name'."
    echo -e "To deactivate the environment, run 'conda deactivate'."
    echo -e "To delete the environment, run 'rm -rf $env_path; rm -rf $symlink_path'."
}

# =================================================================================
# Main
# ================================================================================= 

if [[ $1 == "--help" || $1 == "-h" ]]; then
    usage
    exit 0
fi

# If script is being executed (not sourced), run install with provided arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    install "$@"
fi
