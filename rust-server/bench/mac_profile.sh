#!/usr/bin/env bash
# Per-stage profile of the candle backend (CLI only; no server, no ports). Same 53-word paragraph,
# 1 warm-up + RUNS runs, median. KOKORO_PROFILE=1 syncs the device at every stage boundary.
set -u
cd "$(dirname "$0")/.."
BIN=target/release/kokoro-reader-server; RUNS=${RUNS:-3}; OUT=${OUT:-$HOME/kokoro-rs-profile}; mkdir -p "$OUT"
T="$(cat bench/para53.txt)"
run() { # label env...
  local label=$1; shift
  echo "== $label"
  env "$@" $BIN profile --runs $RUNS --out "$OUT/$label.wav" "$T" 2>"$OUT/$label.err" | tee -a "$OUT/profile.jsonl" | python3 -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l); st=d.pop('stages_s'); print(json.dumps(d))
    for k,v in st.items(): print(f'   {v:8.3f}  {k}')"
  tail -2 "$OUT/$label.err" | grep -iv "^$" | head -2
}
: > "$OUT/profile.jsonl"
run metal-e2e              KOKORO_DEVICE=metal
run metal-e2e-depthwise0   KOKORO_DEVICE=metal KOKORO_DEPTHWISE=0
run metal-prof             KOKORO_DEVICE=metal KOKORO_PROFILE=1
run metal-prof-depthwise0  KOKORO_DEVICE=metal KOKORO_PROFILE=1 KOKORO_DEPTHWISE=0
RUNS=1 run candle-cpu-prof KOKORO_DEVICE=candle-cpu KOKORO_PROFILE=1
run metal-det              KOKORO_DEVICE=metal KOKORO_CANDLE_DETERMINISTIC=1
run metal-det-depthwise0   KOKORO_DEVICE=metal KOKORO_CANDLE_DETERMINISTIC=1 KOKORO_DEPTHWISE=0
python3 - "$OUT/metal-det.wav" "$OUT/metal-det-depthwise0.wav" <<'PY'
import sys, wave, array
a, b = [array.array('h', wave.open(p).readframes(10**9)) for p in sys.argv[1:3]]
print({"det_depthwise1_vs_0": {"len": [len(a), len(b)], "max_abs_diff": max(abs(x-y) for x, y in zip(a, b)) / 32768}})
PY
