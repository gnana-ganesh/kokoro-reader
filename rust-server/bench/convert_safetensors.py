"""Flatten hexgrad/Kokoro-82M kokoro-v1_0.pth into kokoro-v1_0.safetensors (+ config.json) for the
candle backend (KOKORO_DEVICE=metal|candle-cpu). Needs torch + huggingface_hub (the Python server venv has both).
usage: python convert_safetensors.py OUT_DIR"""
import json, os, struct, sys, shutil
import torch
from huggingface_hub import hf_hub_download
out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/kokoro-reader")
pth = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
cfg = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
sd = torch.load(pth, map_location="cpu", weights_only=True)
flat = {}
for part, d in sd.items():
    for k, t in d.items():
        k = k[7:] if k.startswith("module.") else k
        flat[f"{part}.{k}"] = t.detach().float().contiguous()
# minimal safetensors writer (no extra dependency)
header, off, blobs = {}, 0, []
for k in sorted(flat):
    b = flat[k].numpy().tobytes(); header[k] = {"dtype": "F32", "shape": list(flat[k].shape), "data_offsets": [off, off + len(b)]}
    off += len(b); blobs.append(b)
h = json.dumps(header, separators=(",", ":")).encode(); h += b" " * ((8 - len(h) % 8) % 8)
os.makedirs(out, exist_ok=True)
tmp = os.path.join(out, "kokoro-v1_0.safetensors.tmp")
with open(tmp, "wb") as f:
    f.write(struct.pack("<Q", len(h))); f.write(h)
    for b in blobs: f.write(b)
os.replace(tmp, os.path.join(out, "kokoro-v1_0.safetensors"))
shutil.copy(cfg, os.path.join(out, "config.json"))
print(f"wrote {len(flat)} tensors, {off/1e6:.0f} MB to {out}")
