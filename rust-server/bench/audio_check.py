"""Audio sanity: duration, RMS, silent fraction, and (optional) WER vs the source text.
usage: python audio_check.py SOURCE.txt REF.wav OTHER.wav [...]   (stdlib only; WER if faster-whisper+jiwer present)"""
import sys, wave, array, math, re, json
def load(p):
    w = wave.open(p); sr = w.getframerate(); a = array.array('h', w.readframes(w.getnframes())); return sr, a
def stats(p):
    sr, a = load(p); n = len(a); rms = math.sqrt(sum(x*x for x in a)/max(n,1))/32768
    win = sr//50; frames = [a[i:i+win] for i in range(0, n-win, win)]
    silent = sum(1 for f in frames if max(abs(x) for x in f) < 0.01*32768)/max(len(frames),1)
    return {"file": p, "secs": round(n/sr, 2), "rms": round(rms, 4), "silent_frac": round(silent, 3)}
src = open(sys.argv[1]).read()
rows = [stats(p) for p in sys.argv[2:]]
try:
    import warnings; warnings.filterwarnings("ignore")
    from faster_whisper import WhisperModel; import jiwer, numpy as np, soundfile as sf
    from scipy.signal import resample_poly
    m = WhisperModel("small.en", device="cpu", compute_type="int8")
    norm = lambda s: " ".join(re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split())
    for r in rows:
        a, sr = sf.read(r["file"], dtype="float32"); a = resample_poly(a, 2, 3).astype(np.float32)
        segs, _ = m.transcribe(a, beam_size=5, language="en"); hyp = " ".join(s.text.strip() for s in segs)
        r["wer"] = round(jiwer.wer(norm(src), norm(hyp)), 3); r["asr"] = hyp
except ImportError:
    pass
ref = rows[0]["secs"]
for r in rows:
    r["len_vs_ref"] = round(r["secs"]/ref, 3); print(json.dumps(r))
