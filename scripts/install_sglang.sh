#!/bin/bash
# Install SGLang + SGLang Router into a dedicated venv under `src/sglang`.
#
# Usage:
#   bash scripts/install_sglang.sh

set -eo pipefail

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
CWD=$(pwd)

git submodule update --init

SGLANG_DIR="$CWD/src/sglang"
cd "$SGLANG_DIR"

# Fetch KVFlow branch
git checkout --track origin/kvflow || true
git checkout helium

# Create (or reuse) a dedicated venv for SGLang.
uv venv .venv --python "$PYTHON_VERSION"
SGLANG_PYTHON="$SGLANG_DIR/.venv/bin/python"

# Install SGLang with all optional dependencies into the SGLang venv.
uv pip install -p "$SGLANG_PYTHON" -e "python[all]" --index-strategy unsafe-best-match

# Install SGLang router
cd sgl-router
# Install rustup
bash $CWD/scripts/install_rustup.sh
source "$HOME/.cargo/env"

# Install build dependencies into the SGLang venv.
uv pip install -p "$SGLANG_PYTHON" setuptools-rust wheel build
# Build the wheel package
$SGLANG_PYTHON -m build
# Install the generated wheel
uv pip install -p "$SGLANG_PYTHON" dist/*.whl
# One-liner for development (rebuild + install)
# $SGLANG_PYTHON -m build && uv pip install -p "$SGLANG_PYTHON" --force-reinstall dist/*.whl

cd $CWD
