"""Pre-download Kokoro-82M weights + voices into the local Hugging Face cache (one-time, needs internet)."""
from huggingface_hub import hf_hub_download
REPO = "hexgrad/Kokoro-82M"
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_michael", "am_adam",
          "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
for f in ["config.json", "kokoro-v1_0.pth"] + [f"voices/{v}.pt" for v in VOICES]:
    p = hf_hub_download(repo_id=REPO, filename=f)
    print("ok", p)
