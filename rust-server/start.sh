#!/usr/bin/env bash
# Start the Rust Kokoro server (drop-in for ../server.py). Env knobs: see README.md.
set -euo pipefail
cd "$(dirname "$0")"
BIN=target/release/kokoro-reader-server
if [ ! -x "$BIN" ]; then
  if [ "$(uname)" = "Darwin" ]; then cargo build --release --features coreml; else cargo build --release; fi
fi
./download_models.sh "${KOKORO_MODEL_VARIANT:-fp32}" >/dev/null
exec "$BIN"
