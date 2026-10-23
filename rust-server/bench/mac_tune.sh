#!/usr/bin/env bash
# Thread/variant sweep for the Rust server on 8881 (+ one Python run on $PY_PORT). bash only (zsh doesn't word-split).
set -u
cd "$(dirname "$0")"
BIN=../target/release/kokoro-reader-server; PORT=8881; PY_PORT=${PY_PORT:-8880}; RUNS=${RUNS:-3}
OUT=${OUT:-/tmp/kokoro-rs-bench}; mkdir -p "$OUT"; RES="$OUT/tune.jsonl"; : > "$RES"
sysctl -n vm.swapusage 2>/dev/null
run() {
  local L=$1; shift
  env KOKORO_PORT=$PORT "$@" $BIN > "$OUT/$L.log" 2>&1 & local P=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 0.5; done
  python3 bench.py http://127.0.0.1:$PORT $P rs-$L $RUNS para53.txt | tee -a "$RES"
  command -v footprint >/dev/null && footprint -p $P 2>/dev/null | grep "phys_footprint:" | sed "s/^/  $L/"
  kill $P; wait $P 2>/dev/null; sleep 5
}
run int8-t8 KOKORO_MODEL_VARIANT=int8
run int8-t4 KOKORO_MODEL_VARIANT=int8 KOKORO_THREADS=4
run fp32-t8 KOKORO_MODEL_VARIANT=fp32
run fp32-t4 KOKORO_MODEL_VARIANT=fp32 KOKORO_THREADS=4
run fp16-t4 KOKORO_MODEL_VARIANT=fp16 KOKORO_THREADS=4
run int8-t4-noarena KOKORO_MODEL_VARIANT=int8 KOKORO_THREADS=4 KOKORO_CPU_ARENA=0
PY=$(lsof -ti tcp:$PY_PORT -sTCP:LISTEN | head -1)
python3 bench.py http://127.0.0.1:$PY_PORT "${PY:-0}" python-$PY_PORT $RUNS para53.txt | tee -a "$RES"
sysctl -n vm.swapusage 2>/dev/null
