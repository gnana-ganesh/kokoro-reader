#!/bin/bash
# Start the Kokoro TTS server.
#  - If the LaunchAgent is installed (./install-launchagent.sh), it is started via launchd.
#  - Otherwise the server is started in the background with nohup.
#  - ./start.sh --foreground runs it in this terminal (useful for debugging).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${KOKORO_LABEL:-com.kokoro-reader.server}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PORT="${KOKORO_PORT:-8880}"
PY="$DIR/.venv/bin/python"
LOG="$DIR/server.log"
export PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 KOKORO_DEVICE="${KOKORO_DEVICE:-mps}"

if [ ! -x "$PY" ]; then
  echo "No virtualenv at $DIR/.venv. See README.md (Install)."; exit 1
fi

if [ "$1" = "--foreground" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
  cd "$DIR" && exec "$PY" "$DIR/server.py"
fi

if [ -f "$PLIST" ]; then
  if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
    launchctl kickstart "gui/$(id -u)/$LABEL"
  else
    for i in 1 2 3 4 5; do
      launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null && break
      sleep 1
    done
  fi
elif ! curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  cd "$DIR" && nohup "$PY" "$DIR/server.py" >>"$LOG" 2>&1 &
fi

echo -n "Waiting for Kokoro TTS on http://127.0.0.1:$PORT "
for i in $(seq 1 90); do
  if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo; echo "Running: $(curl -s "http://127.0.0.1:$PORT/health")"; exit 0
  fi
  echo -n "."; sleep 1
done
echo; echo "Server did not come up within 90s. See $LOG"; tail -20 "$LOG"; exit 1
