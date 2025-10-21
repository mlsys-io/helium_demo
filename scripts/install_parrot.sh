#!/bin/bash
# Fetch ParrotServe submodule and install it in editable mode
#
# Usage:
#   bash scripts/install_parrot.sh

set -eo pipefail

CUDA_VERSION="cuda-12.6"
export TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME="/usr/local/$CUDA_VERSION"
export LD_LIBRARY_PATH="/usr/local/$CUDA_VERSION/lib64:$LD_LIBRARY_PATH"

git submodule update --init
cd src/ParrotServe
git checkout helium
uv pip install parse
uv pip install -e . --no-build-isolation
cd -
