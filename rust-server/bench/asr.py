import warnings; warnings.filterwarnings("ignore")
from faster_whisper import WhisperModel
import jiwer, re, soundfile as sf, numpy as np
from scipy.signal import resample_poly
def load(p):
    a, sr = sf.read(p, dtype="float32"); return resample_poly(a, 2, 3).astype(np.float32)
m = WhisperModel("small.en", device="cpu", compute_type="int8")
src = open('tricky.txt').read().splitlines()
def norm(s):
    s = s.lower(); s = re.sub(r"[^a-z0-9' ]+", " ", s); return " ".join(s.split())
res = {}
for label in ["py", "rs-loqui", "rs-espeak", "rs-misaki-rs"]:
    hyps = []
    for i in range(len(src)):
        segs, _ = m.transcribe(load(f"asr/{label}_{i}.wav"), beam_size=5, language="en")
        hyps.append(" ".join(s.text.strip() for s in segs))
    res[label] = hyps
for i, s in enumerate(src):
    print("\nTXT", s)
    for k, v in res.items(): print(f"  {k:12}", v[i])
print()
ref = res["py"]
for k, v in res.items():
    print(f"{k:12} WER vs source text: {jiwer.wer([norm(x) for x in src], [norm(x) for x in v]):.3f}   WER vs Python-server transcript: {jiwer.wer([norm(x) for x in ref], [norm(x) for x in v]):.3f}")
