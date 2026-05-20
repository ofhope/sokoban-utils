#!/usr/bin/env bash
# First-time setup: install Rust, wasm-target, and wasm-pack.
# Safe to re-run — each step is idempotent.
set -euo pipefail

echo "── Checking Rust ──"
if ! command -v cargo &>/dev/null; then
  echo "Installing Rust via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
else
  echo "Rust $(rustc --version) already installed."
fi

echo "── Checking wasm32 target ──"
if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
  rustup target add wasm32-unknown-unknown
else
  echo "wasm32-unknown-unknown already installed."
fi

echo "── Checking wasm-pack ──"
if ! command -v wasm-pack &>/dev/null; then
  echo "Installing wasm-pack..."
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
else
  echo "wasm-pack $(wasm-pack --version) already installed."
fi

echo ""
echo "Setup complete. Run 'npm run build' to compile."
