#!/bin/bash
# Stop the Kokoro TTS server. If the LaunchAgent is installed it starts again at next login (or run ./start.sh).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${KOKORO_LABEL:-com.kokoro-reader.server}"
if launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null; then
  for i in $(seq 1 20); do launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || break; sleep 0.5; done
  echo "Stopped (launchd)."
else
  echo "Was not running under launchd."
fi
pkill -f "$DIR/server.py" 2>/dev/null && echo "Stopped background/foreground server." || true
