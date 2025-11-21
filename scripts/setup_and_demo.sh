#!/usr/bin/env bash
set -euo pipefail

# One-shot setup and Rich TUI demo runner.
# - Ensures nightly Rust (for portable_simd)
# - Builds fastcma with maturin
# - Creates a uv-managed venv (Python 3.13)
# - Installs demo extras (rich) and runs the TUI demo

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/5] Checking rustup toolchain (nightly)..."
if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup not found. Please install Rust (https://rustup.rs) and re-run." >&2
  exit 1
fi
rustup toolchain install nightly --component rustfmt clippy >/dev/null 2>&1 || true
rustup override set nightly >/dev/null 2>&1

echo "[2/5] Ensuring uv is available..."
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install from https://github.com/astral-sh/uv/releases and re-run." >&2
  exit 1
fi

echo "[3/5] Creating Python 3.13 venv with uv..."
uv venv --clear --python 3.13 .venv

echo "[4/5] Installing build/deps (maturin + demo extras) ..."
uv pip install --upgrade pip
uv pip install maturin
uv pip install .[demo]

echo "[5/5] Building extension (maturin develop --release)..."
uv run maturin develop --release

echo "Running Rich TUI demo..."
uv run python examples/rich_tui_demo.py
