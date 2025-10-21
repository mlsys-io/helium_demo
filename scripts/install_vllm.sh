#!/bin/bash
# Fetch vLLM submodule and install it in editable mode
#
# Usage:
#   bash scripts/install_vllm.sh

set -eo pipefail

git submodule update --init
cd src/vllm
git checkout v0.8.5
export VLLM_COMMIT=8fc88d63f1163f119dd740b1666069535f052ff3 # Commit hash for vLLM version 0.8.5
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
uv pip install -e .
cd -
