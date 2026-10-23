# Kokoro Reader

**Offline, local text-to-speech for any webpage in Chrome, powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).**

Kokoro Reader is a small Chrome extension plus a local TTS server. Click the toolbar icon on an article, docs page or
online book and it reads the main text aloud in a natural-sounding voice, highlighting each paragraph as it goes.
Everything runs on your own machine: no cloud TTS, no accounts, no text leaves your computer.

It was originally built for reading O'Reilly books on [learning.oreilly.com](https://learning.oreilly.com) and works on
most other sites too (Wikipedia, MDN, GitHub READMEs, blogs, ...).

## Features
- **Natural, offline voices**: 10 English voices (US and UK) from Kokoro-82M, synthesized locally on the Apple GPU (MPS) or CPU.
- **Reads any webpage**: the main article text is detected with Mozilla Readability (with fallbacks), so menus and sidebars are skipped.
- **Paragraph highlighting**: the current paragraph is highlighted and scrolled into view; the next one is synthesized ahead of time for gapless playback.
- **Controls**: play/pause, stop, previous/next paragraph, voice picker, speed (0.8x to 1.5x), draggable panel. Settings are remembered.
- **Start anywhere**: Option+click (Alt+click) any paragraph to read from there.
- **Read a selection**: select text and right-click **Read aloud with Kokoro**, or press play with text selected.
- **Skip code blocks** (on by default, toggle in the panel), handy for technical books.
- **Unobtrusive**: on most sites nothing runs until you click the icon; on learning.oreilly.com the panel appears automatically.
- **Locked-down server**: listens on `127.0.0.1` only and rejects browser requests from any origin other than a Chrome extension, so websites can't use it.

## Requirements
- macOS on Apple Silicon (M1 or newer). Other platforms may work with `KOKORO_DEVICE=cpu`, but only macOS is tested and the scripts use `launchctl`.
- [Homebrew](https://brew.sh)
- Python 3.12
- Google Chrome (or another Chromium browser that supports Manifest V3 extensions)
- About 2.5 GB of disk (mostly PyTorch) and about 2 GB of free RAM while the server runs (about 1.2 GB on CPU)

## Install
```bash
git clone https://github.com/gnana-ganesh/kokoro-reader.git
cd kokoro-reader

# 1. Phonemizer fallback used by Kokoro
brew install espeak-ng

# 2. Python environment
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 3. Download the model and voices into the Hugging Face cache (one-time, needs internet)
.venv/bin/python download_models.py

# 4. Start the server
./start.sh
curl http://127.0.0.1:8880/health
```

### Optional: start the server automatically at login
```bash
./install-launchagent.sh                     # uses the Apple GPU (mps)
KOKORO_DEVICE=cpu ./install-launchagent.sh   # CPU only: less memory, roughly 2x slower
./install-launchagent.sh --uninstall         # remove it
```
This writes `~/Library/LaunchAgents/com.kokoro-reader.server.plist` from
[`launchd/com.kokoro-reader.server.plist.template`](launchd/com.kokoro-reader.server.plist.template), filling in this folder's path.

### Load the Chrome extension
1. Open `chrome://extensions` and turn on **Developer mode** (top right).
2. Click **Load unpacked** and pick the `extension/` folder in this repo (in the macOS file dialog, Cmd+Shift+G lets you paste a path).
3. Pin **Kokoro Reader** from the puzzle-piece menu so its icon is in the toolbar.
4. Reload any tabs that were already open.

After pulling updates, click the reload arrow on Kokoro Reader in `chrome://extensions` and reload your tabs.

## Usage
- **Any webpage**: click the Kokoro Reader toolbar icon (or press **Option+Shift+P**) to show the panel, then press **▶**.
  Reading starts at the first visible paragraph.
- **O'Reilly books**: open a chapter on learning.oreilly.com; the panel appears automatically in the bottom-right.
- **Option+click** (Alt+click) any paragraph to start reading from there.
- **Selected text**: select it and press ▶, or right-click → **Read aloud with Kokoro**.
- Click the toolbar icon again (or ✕ on the panel) to hide the panel.

| Control | Action |
|---|---|
| ▶ / ⏸ | Play / pause |
| ⏹ | Stop |
| ⏮ / ⏭ | Previous / next paragraph |
| Option+Shift+P | Play / pause from the keyboard (change it at `chrome://extensions/shortcuts`) |
| Option+click | Read from that paragraph |

## Server
`server.py` is a small FastAPI app on `http://127.0.0.1:8880`:

| Endpoint | Description |
|---|---|
| `GET /health` | Status, device, number of voices |
| `GET /voices` | Available voices and the default |
| `POST /tts` | `{"text": "...", "voice": "af_heart", "speed": 1.0}` → `audio/wav` |

```bash
curl -s -X POST http://127.0.0.1:8880/tts -H 'Content-Type: application/json' \
     -d '{"text":"Hello from Kokoro.","voice":"af_heart","speed":1.0}' -o /tmp/k.wav && afplay /tmp/k.wav
```

Environment variables: `KOKORO_DEVICE` (`mps` or `cpu`; `start.sh` and the LaunchAgent default to `mps`),
`KOKORO_PORT` (default `8880`, the extension expects 8880), `KOKORO_DEFAULT_VOICE` (default `af_heart`), `KOKORO_THREADS`.

Scripts: `./start.sh` (add `--foreground` to run in the terminal for debugging), `./stop.sh`. Logs go to `server.log`.

**Voices:** af_heart, af_bella, af_nicole, af_sarah, am_michael, am_adam (US); bf_emma, bf_isabella, bm_george, bm_lewis (UK).
To add more, add their names to `download_models.py`, run it again, then `./stop.sh && ./start.sh`.

On an M-series MacBook Air a 60-word paragraph takes about 2.5 s to synthesize on the GPU, far faster than it takes to play.

## Troubleshooting
- **Panel says the server is offline**: run `./start.sh` and check `curl http://127.0.0.1:8880/health`. Look at `server.log` for errors.
- **Nothing happens when clicking the icon**: extensions can't run on Chrome's own pages (`chrome://`, the Web Store, the new-tab page).
  For local `file://` pages, enable "Allow access to file URLs" in the extension's Details.
- **It skips the first paragraph or picks the wrong text** on some sites: Option+click the paragraph you want to start from.
- **Server fails to start offline**: the model must be downloaded once with `download_models.py` (the server runs with `HF_HUB_OFFLINE=1`).
- **High memory use**: switch to CPU with `KOKORO_DEVICE=cpu ./install-launchagent.sh` (or `KOKORO_DEVICE=cpu ./start.sh`).
- **403 "origin not allowed"**: by design, only Chrome extensions (and non-browser clients such as curl) may call the server.

## Privacy
All text-to-speech happens locally. The extension only talks to `http://127.0.0.1:8880`; the server binds to localhost,
never contacts the network at runtime (`HF_HUB_OFFLINE=1`) and does not store the text it reads.
The only network access is the one-time install and model download.

## Uninstall
```bash
./install-launchagent.sh --uninstall   # if you installed the LaunchAgent
./stop.sh
rm -rf ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
```
Then remove the extension at `chrome://extensions` and delete this folder.

## Credits
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and the [`kokoro`](https://github.com/hexgrad/kokoro) library by hexgrad, Apache-2.0.
- [Mozilla Readability](https://github.com/mozilla/readability), Apache-2.0, vendored as `extension/Readability.js`
  (license in [`extension/LICENSE-Readability.md`](extension/LICENSE-Readability.md)).

## License
[MIT](LICENSE) © 2026 Gnana Ganesh. Third-party components keep their own licenses (see Credits).
