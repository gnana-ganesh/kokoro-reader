#!/usr/bin/env bash
# Side-by-side benchmark on macOS: Rust server on 8881 (several configs) vs the running Python
# server on 8880 (only read/POST /tts requests; it is never restarted). Results -> $OUT/results.jsonl
set -u
cd "$(dirname "$0")/.."
BIN=target/release/kokoro-reader-server
PORT=8881; PY_PORT=${PY_PORT:-8880}; RUNS=${RUNS:-3}; OUT=${OUT:-/tmp/kokoro-rs-bench}; mkdir -p "$OUT"
: > "$OUT/results.jsonl"
cd bench
start_rs() {  # $1 label, rest env assignments
  local label=$1; shift
  local t0=$(python3 -c 'import time;print(time.time())')
  env KOKORO_PORT=$PORT "$@" ../$BIN > "$OUT/$label.log" 2>&1 &
  RS_PID=$!
  for i in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $RS_PID 2>/dev/null || { echo "  $label: exited"; tail -5 "$OUT/$label.log"; RS_PID=; return 1; }
    sleep 0.5
  done
  STARTUP=$(python3 -c "import time;print(round(time.time()-$t0,1))")
  echo "  $label: ready in ${STARTUP}s (pid $RS_PID)"
}
stop_rs() { [ -n "${RS_PID:-}" ] && kill $RS_PID 2>/dev/null; wait $RS_PID 2>/dev/null; RS_PID=; sleep 1; }
trap stop_rs EXIT

run_cfg() {  # label env...
  local label=$1
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $label "* ]] && return
  if start_rs "$@"; then
    r=$(python3 bench.py http://127.0.0.1:$PORT $RS_PID $label $RUNS para53.txt 2>&1 | tail -1)
    echo "$r" | python3 -c "import sys,json;d=json.loads(sys.stdin.read());d['startup_s']=$STARTUP;print(json.dumps(d))" >> "$OUT/results.jsonl" || echo "$r" >> "$OUT/results.jsonl"
    tail -1 "$OUT/results.jsonl"
    grep -iE "warn|error|fallback|partition|nodes" "$OUT/$label.log" | head -5
    stop_rs
  fi
}

echo "== parity vs Python server on 8880 (CPU fp32 Rust)"
if start_rs parity; then python3 parity.py http://127.0.0.1:$PY_PORT http://127.0.0.1:$PORT > "$OUT/parity.txt" 2>&1; tail -2 "$OUT/parity.txt"; grep -E "^(DIFF|FAIL)" "$OUT/parity.txt" | head; stop_rs; fi

echo "== Python server on $PY_PORT"
PY_PID=$(lsof -ti tcp:$PY_PORT -sTCP:LISTEN | head -1)
python3 bench.py http://127.0.0.1:$PY_PORT "${PY_PID:-0}" python-$PY_PORT $RUNS para53.txt | tee -a "$OUT/results.jsonl"

echo "== Rust configs"
run_cfg rs-cpu-fp32 KOKORO_DEVICE=cpu KOKORO_MODEL_VARIANT=fp32
run_cfg rs-cpu-fp16 KOKORO_DEVICE=cpu KOKORO_MODEL_VARIANT=fp16
run_cfg rs-cpu-int8 KOKORO_DEVICE=cpu KOKORO_MODEL_VARIANT=int8
run_cfg rs-coreml-all-fp32 KOKORO_DEVICE=coreml KOKORO_COREML_UNITS=all KOKORO_MODEL_VARIANT=fp32
run_cfg rs-coreml-all-fp32-cached KOKORO_DEVICE=coreml KOKORO_COREML_UNITS=all KOKORO_MODEL_VARIANT=fp32
run_cfg rs-coreml-gpu-fp32 KOKORO_DEVICE=coreml KOKORO_COREML_UNITS=gpu KOKORO_MODEL_VARIANT=fp32
run_cfg rs-coreml-ane-fp32 KOKORO_DEVICE=coreml KOKORO_COREML_UNITS=ane KOKORO_MODEL_VARIANT=fp32
run_cfg rs-coreml-nn-fp32 KOKORO_DEVICE=coreml KOKORO_COREML_FORMAT=nn KOKORO_MODEL_VARIANT=fp32
echo "== Python after (still healthy?)"; curl -s http://127.0.0.1:$PY_PORT/health; echo
