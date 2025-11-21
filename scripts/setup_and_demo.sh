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
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "[3/5] Creating Python 3.13 venv with uv..."
uv venv --clear --python 3.13 .venv

echo "[4/5] Installing build/deps (maturin + demo extras) ..."
uv pip install --upgrade pip
uv pip install maturin rich python-decouple

echo "[5/5] Building wheel (maturin build --release)..."
uv run maturin build --release --out dist
echo "Installing built wheel..."
uv run python -m pip install dist/*.whl

# Some installers leave only an editable .pth; ensure the extension files are present on sys.path.
export WHEEL_PATH=$(ls dist/*.whl | head -n1)
uv run env WHEEL_PATH="$WHEEL_PATH" python - <<'PY'
import sys, os, zipfile
wheel = os.environ["WHEEL_PATH"]
site = next(p for p in sys.path if "site-packages" in p)
with zipfile.ZipFile(wheel) as z:
    z.extractall(site)
print("Extracted wheel contents to", site)
PY

# Ensure the freshly built native module is on PYTHONPATH (works even if pip left an editable .pth)
export PYTHONPATH="$PROJECT_ROOT/target/release:$PROJECT_ROOT/python:${PYTHONPATH:-}"

echo "Running Rich TUI demo..."
uv run env PYTHONPATH="$PYTHONPATH" python examples/rich_tui_demo.py
