#!/bin/bash
# Install (or remove) a per-user LaunchAgent that starts the Kokoro TTS server at login.
#   ./install-launchagent.sh              install, using the Apple GPU (mps)
#   KOKORO_DEVICE=cpu ./install-launchagent.sh   install, CPU only (less memory, slower)
#   ./install-launchagent.sh --uninstall  remove it
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${KOKORO_LABEL:-com.kokoro-reader.server}"
DEVICE="${KOKORO_DEVICE:-mps}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "$1" = "--uninstall" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $PLIST"; exit 0
fi

[ -x "$DIR/.venv/bin/python" ] || { echo "No virtualenv at $DIR/.venv. Follow the Install steps in README.md first."; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents"
sed -e "s|__LABEL__|$LABEL|g" -e "s|__DIR__|$DIR|g" -e "s|__DEVICE__|$DEVICE|g" \
  "$DIR/launchd/com.kokoro-reader.server.plist.template" > "$PLIST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed $PLIST (device: $DEVICE). The server now starts at login."
"$DIR/start.sh"
