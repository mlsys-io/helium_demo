#!/bin/bash
# Fetch LMCache submodule and install it in editable mode
#
# Usage:
#   bash scripts/install_lmcache.sh

set -eo pipefail

git submodule update --init
cd src/LMCache
git checkout v0.3.5
uv pip install -r requirements/build.txt
uv pip install -e . --no-build-isolation
cd -