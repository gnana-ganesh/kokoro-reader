#!/usr/bin/env bash
# Mac call 1 (GPU paths): fast-forward the worktree from a bundle, build with coreml+webgpu+candle,
# convert hexgrad weights to safetensors for the candle/Metal backend. Never touches port 8880.
set -eu
BUNDLE=${1:-$HOME/rust-server-gpu.bundle}
cd "$HOME/kokoro-reader-rust"
git fetch "$BUNDLE" rust-server && git merge --ff-only FETCH_HEAD && git log --oneline -1
cd rust-server
echo "disk before:"; df -h "$HOME" | tail -1
t0=$(date +%s)
cargo build --release --features coreml,webgpu,candle 2>&1 | grep -E "^(error|warning: unused)|-->|Finished" | head -20
echo "build: $(( $(date +%s) - t0 ))s"; ls -la target/release/kokoro-reader-server
DIR=$HOME/.cache/kokoro-reader
if [ ! -f "$DIR/kokoro-v1_0.safetensors" ]; then
  PY=""
  for p in "$HOME/kokoro-reader/.venv/bin/python" "$HOME/kokoro-reader/venv/bin/python"; do
    [ -x "$p" ] && "$p" -c "import torch, huggingface_hub" 2>/dev/null && PY=$p && break
  done
  [ -n "$PY" ] || { echo "no python with torch found"; exit 1; }
  "$PY" bench/convert_safetensors.py "$DIR"
fi
ls -la "$DIR"; du -sh target "$DIR"; echo "disk after:"; df -h "$HOME" | tail -1
