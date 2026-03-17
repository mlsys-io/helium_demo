#!/bin/bash
# Install rustup for build tools
#
# Usage:
#   bash scripts/install_rustup.sh

set -eo pipefail

# Source rustup environment if it exists
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Check rustc and cargo installation
if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
    echo "rustc and cargo are already installed."
    rustc --version
    cargo --version
    exit 0
fi

# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the installation prompts, then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
