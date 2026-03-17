#!/bin/bash
# Fetch vLLM submodule and install it in editable mode
#
# Usage:
#   bash scripts/install_vllm.sh

set -eo pipefail

git submodule update --init
cd src/vllm
git checkout helium-v0.16.0
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=89a77b10846fd96273cce78d86d2556ea582d26e # Upstream v0.16.0 wheel commit
uv pip install -e .
cd -
