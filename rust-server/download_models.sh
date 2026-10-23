#!/usr/bin/env bash
# Download Kokoro-82M v1.0 ONNX weights + voices (from thewh1teagle/kokoro-onnx releases, MIT;
# model weights are hexgrad/Kokoro-82M, Apache-2.0).
# Usage: ./download_models.sh [fp32|fp16|int8|all]   (default: fp32)
set -euo pipefail
DIR="${KOKORO_MODEL_DIR:-$HOME/.cache/kokoro-reader}"
BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
WHICH="${1:-fp32}"
mkdir -p "$DIR"
files=(voices-v1.0.bin)
case "$WHICH" in
  fp32) files+=(kokoro-v1.0.onnx) ;;
  fp16) files+=(kokoro-v1.0.fp16.onnx) ;;
  int8) files+=(kokoro-v1.0.int8.onnx) ;;
  all)  files+=(kokoro-v1.0.onnx kokoro-v1.0.fp16.onnx kokoro-v1.0.int8.onnx) ;;
  *) echo "unknown variant $WHICH" >&2; exit 1 ;;
esac
for f in "${files[@]}"; do
  if [ -s "$DIR/$f" ]; then echo "ok   $DIR/$f"; continue; fi
  echo "get  $f"
  curl -fL --retry 3 -o "$DIR/$f.part" "$BASE/$f" && mv "$DIR/$f.part" "$DIR/$f"
done
echo "Models in $DIR"
