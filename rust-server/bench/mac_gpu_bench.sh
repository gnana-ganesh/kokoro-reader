#!/usr/bin/env bash
# Mac call 2 (GPU paths): same methodology as mac_bench.sh (para53.txt, 1 warm-up + RUNS, median,
# RSS + footprint, first-call time), Rust on 8881 only; the Python server on 8880 only gets /tts POSTs.
set -u
cd "$(dirname "$0")/.."
BIN=target/release/kokoro-reader-server
PORT=8881; PY_PORT=${PY_PORT:-8880}; RUNS=${RUNS:-3}; OUT=${OUT:-$HOME/kokoro-rs-gpu-bench}; mkdir -p "$OUT"
: > "$OUT/results.jsonl"; : > "$OUT/footprint.txt"
cd bench
start_rs() {
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
  echo "  $label: ready in ${STARTUP}s (pid $RS_PID) $(curl -s http://127.0.0.1:$PORT/health)"
}
stop_rs() { [ -n "${RS_PID:-}" ] && kill $RS_PID 2>/dev/null; wait $RS_PID 2>/dev/null; RS_PID=; sleep 1; }
trap stop_rs EXIT
run_cfg() {
  local label=$1
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $label "* ]] && return
  if start_rs "$@"; then
    r=$(python3 bench.py http://127.0.0.1:$PORT $RS_PID $label $RUNS para53.txt 2>&1 | tail -1)
    echo "$r" | python3 -c "import sys,json;d=json.loads(sys.stdin.read());d['startup_s']=$STARTUP;print(json.dumps(d))" >> "$OUT/results.jsonl" || echo "{\"label\":\"$label\",\"error\":$(printf %s "$r" | python3 -c 'import sys,json;print(json.dumps(sys.stdin.read()[-300:]))')}" >> "$OUT/results.jsonl"
    tail -1 "$OUT/results.jsonl"
    footprint -p $RS_PID 2>/dev/null | grep -iE "footprint:" | head -1 | sed "s/^/  $label /" | tee -a "$OUT/footprint.txt"
    grep -iE "warn|error|fallback|fail" "$OUT/$label.log" | head -4
    stop_rs
  fi
}
if [ -z "${SKIP_PY:-}" ]; then
echo "== Python server on $PY_PORT (read-only: POST /tts)"
PY_PID=$(lsof -ti tcp:$PY_PORT -sTCP:LISTEN | head -1)
python3 bench.py http://127.0.0.1:$PY_PORT "${PY_PID:-0}" python-$PY_PORT $RUNS para53.txt | tee -a "$OUT/results.jsonl"
footprint -p ${PY_PID:-0} 2>/dev/null | grep -iE "footprint:" | head -1 | sed "s/^/  python /" | tee -a "$OUT/footprint.txt"
fi
echo "== Rust configs"
run_cfg rs-cpu-int8-t4      KOKORO_DEVICE=cpu KOKORO_MODEL_VARIANT=int8 KOKORO_THREADS=4
run_cfg rs-metal            KOKORO_DEVICE=metal
run_cfg rs-metal-2          KOKORO_DEVICE=metal
run_cfg rs-webgpu-fp32      KOKORO_DEVICE=webgpu KOKORO_MODEL_VARIANT=fp32
run_cfg rs-webgpu-fp16      KOKORO_DEVICE=webgpu KOKORO_MODEL_VARIANT=fp16
run_cfg rs-coreml-static    KOKORO_DEVICE=coreml KOKORO_COREML_STATIC=1 KOKORO_MODEL_VARIANT=fp32
run_cfg rs-coreml-static-gpu KOKORO_DEVICE=coreml KOKORO_COREML_STATIC=1 KOKORO_COREML_UNITS=gpu KOKORO_MODEL_VARIANT=fp32
RUNS=1 run_cfg rs-metal-candle-lstm KOKORO_DEVICE=metal KOKORO_LSTM=candle
echo "== audio sanity vs rs-cpu-int8-t4"
ls out_*.wav >/dev/null 2>&1 && python3 audio_cmp.py out_rs-cpu-int8-t4.wav $(ls out_*.wav | grep -v int8-t4) | tee "$OUT/audio.jsonl"
echo "== deterministic metal vs candle-cpu (same short text)"
S="The borrow checker validates references at compile time."
KOKORO_CANDLE_DETERMINISTIC=1 KOKORO_DEVICE=metal ../$BIN synth --out "$OUT/det_metal.wav" "$S" 2>&1 | tail -1
KOKORO_CANDLE_DETERMINISTIC=1 KOKORO_DEVICE=candle-cpu ../$BIN synth --out "$OUT/det_cpu.wav" "$S" 2>&1 | tail -1
python3 - "$OUT/det_cpu.wav" "$OUT/det_metal.wav" <<'PY' | tee -a "$OUT/audio.jsonl"
import sys, wave, array, math
a, b = [array.array('h', wave.open(p).readframes(10**9)) for p in sys.argv[1:3]]
n = min(len(a), len(b)); ma, mb = sum(a[:n])/n, sum(b[:n])/n
c = sum((x-ma)*(y-mb) for x, y in zip(a, b)) / math.sqrt(sum((x-ma)**2 for x in a[:n]) * sum((y-mb)**2 for y in b[:n]))
print({"det_cpu_vs_metal_sample_corr": round(c, 4), "len_cpu": len(a), "len_metal": len(b), "max_abs_diff": max(abs(x-y) for x, y in zip(a, b))/32768})
PY
mv out_*.wav "$OUT/" 2>/dev/null
echo "== Python after (still healthy?)"; curl -s http://127.0.0.1:$PY_PORT/health; echo
