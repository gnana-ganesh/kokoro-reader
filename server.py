"""
Local Kokoro-82M TTS server for the Kokoro Reader Chrome extension.
Listens on 127.0.0.1:8880 only.

  GET  /health  -> {"status":"ok", ...}
  GET  /voices  -> {"voices":[...], "default":"af_heart"}
  POST /tts     {"text": "...", "voice": "af_heart", "speed": 1.0}  -> audio/wav
"""
import os

# Must be set before torch / huggingface_hub are imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never hit the network at runtime
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import io
import re
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from kokoro import KModel, KPipeline

REPO_ID = "hexgrad/Kokoro-82M"
HOST, PORT = "127.0.0.1", int(os.environ.get("KOKORO_PORT", "8880"))
SAMPLE_RATE = 24000
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "af_heart")
MAX_CHARS = 6000

# Device: "cpu" (default, lowest memory, very stable) or "mps" (Apple GPU).
DEVICE = os.environ.get("KOKORO_DEVICE", "cpu").lower()
if DEVICE == "mps" and not torch.backends.mps.is_available():
    DEVICE = "cpu"
THREADS = int(os.environ.get("KOKORO_THREADS", "0"))
if THREADS > 0:
    torch.set_num_threads(THREADS)


def _cached_voices():
    """Voices whose .pt files are already in the local HF cache (usable offline)."""
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snap = base / ("models--" + REPO_ID.replace("/", "--")) / "snapshots"
    names = set()
    if snap.exists():
        for p in snap.glob("*/voices/*.pt"):
            names.add(p.stem)
    return sorted(names)


VOICE_LABELS = {
    "af_heart": "Heart (US female)", "af_bella": "Bella (US female)",
    "af_nicole": "Nicole (US female, soft)", "af_sarah": "Sarah (US female)",
    "am_michael": "Michael (US male)", "am_adam": "Adam (US male)",
    "bf_emma": "Emma (UK female)", "bf_isabella": "Isabella (UK female)",
    "bm_george": "George (UK male)", "bm_lewis": "Lewis (UK male)",
}

print(f"[kokoro] loading model on {DEVICE} ...", flush=True)
t0 = time.time()
MODEL = KModel(repo_id=REPO_ID).to(DEVICE).eval()
# One shared model, two G2P front-ends (American 'a' and British 'b').
PIPELINES = {
    "a": KPipeline(lang_code="a", repo_id=REPO_ID, model=MODEL),
    "b": KPipeline(lang_code="b", repo_id=REPO_ID, model=MODEL),
}
LOCK = threading.Lock()   # the model is not re-entrant; serialise synthesis
VOICES = _cached_voices() or [DEFAULT_VOICE]
print(f"[kokoro] ready in {time.time() - t0:.1f}s; voices: {', '.join(VOICES)}", flush=True)


def _clean(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS]


def synthesize(text: str, voice: str, speed: float) -> np.ndarray:
    pipeline = PIPELINES["b" if voice.startswith("b") else "a"]
    chunks = []
    with LOCK, torch.inference_mode():
        for result in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            audio = result.audio
            if audio is None:
                continue
            chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
    if DEVICE == "mps":
        torch.mps.empty_cache()   # hand cached GPU memory back to macOS
    if not chunks:
        return np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
    return np.concatenate(chunks)


# Warm-up so the first real request is fast.
try:
    synthesize("Ready.", DEFAULT_VOICE, 1.0)
except Exception as e:  # pragma: no cover
    print(f"[kokoro] warm-up failed: {e}", flush=True)

# Only the Chrome extension (its service worker) may call us from a browser.
ALLOWED_ORIGIN_RE = r"^chrome-extension://[a-z]{32}$"

app = FastAPI(title="Kokoro local TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOWED_ORIGIN_RE,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Audio-Duration", "X-Synth-Seconds"],
)


@app.middleware("http")
async def origin_guard(request: Request, call_next):
    # Refuse any browser request whose Origin is not this extension (e.g. a random website
    # POSTing to localhost). Requests without an Origin header (curl, the extension's
    # service-worker GETs) are allowed.
    origin = request.headers.get("origin")
    if origin and not re.match(ALLOWED_ORIGIN_RE, origin):
        return JSONResponse({"detail": "origin not allowed"}, status_code=403)
    return await call_next(request)


class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    speed: float = Field(1.0, ge=0.5, le=2.0)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "voices": len(VOICES), "sample_rate": SAMPLE_RATE}


@app.get("/voices")
def voices():
    return {"default": DEFAULT_VOICE,
            "voices": [{"id": v, "label": VOICE_LABELS.get(v, v)} for v in VOICES]}


@app.post("/tts")
def tts(req: TTSRequest):
    text = _clean(req.text)
    if not text:
        raise HTTPException(400, "empty text")
    if req.voice not in VOICES:
        raise HTTPException(400, f"unknown voice {req.voice!r}; available: {VOICES}")
    t = time.time()
    try:
        audio = synthesize(text, req.voice, req.speed)
    except Exception as e:
        raise HTTPException(500, f"synthesis failed: {e}")
    took = time.time() - t
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    dur = len(audio) / SAMPLE_RATE
    print(f"[kokoro] {len(text.split())} words, {req.voice} x{req.speed}: {dur:.1f}s audio in {took:.2f}s", flush=True)
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Audio-Duration": f"{dur:.2f}", "X-Synth-Seconds": f"{took:.2f}",
                             "Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", workers=1)
