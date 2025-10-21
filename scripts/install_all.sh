#!/bin/bash
# Install all dependencies
#
# Usage:
#   bash scripts/install_all.sh

set -eo pipefail

# If .venv exists, warn and exit
if [ -d ".venv" ]; then
    echo "A virtual environment already exists in .venv. Please remove it before running this script."
    exit 1
fi

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install all dependencies managed by uv
echo "Installing uv dependencies..."
uv sync --all-groups

# Build and install vLLM from source
echo "Installing vLLM..."
bash scripts/install_vllm.sh

# Build and install LMCache from source
echo "Installing LMCache..."
bash scripts/install_lmcache.sh

# Build and install ParrotServe from source
echo "Installing ParrotServe..."
bash scripts/install_parrot.sh

echo "All dependencies installed successfully."
